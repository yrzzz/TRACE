"""Main-method training and evaluation; no model or loss registry."""
import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.skin_cells import SkinCellsDataset
from datasets.xenium import XeniumDataset
from losses.coop_moe_loss import compute_coop_moe_loss
from models.dino_tokens import load_dino_tokens, get_dino_dim, build_dino_transform
from models.tri_expert import CellExpertModelE1AttnE2E3Attn
from models.uni_tokens import load_uni_tokens, get_uni_dim
from utils.io import build_batch
from utils.losses import ce_smooth
from utils.metrics import compute_auc_metrics
from utils.diagnostics import compute_metrics_binary, compute_metrics_multiclass
from utils.seed import set_seed
from xenium_prepare import canonicalize_xenium_label_col


DEFAULTS = dict(
    dataset="xenium", cell_size=256, ctx_size=1024, ctx_dino_input=256,
    proj_dim=512, head_hidden_dim=256, head_layers=1, gate_hidden_dim=256,
    gate_layers=2, dropout=0.1, e2_attn_tau=2.0, e3_attn_tau=1.0,
    uni_model_name="hf-hub:MahmoodLab/UNI2-h",
    dino_model_name="vit_base_patch16_dinov3.lvd1689m",
    batch_size=512, num_workers=4, epochs=30, eval_every=1, lr=1e-3,
    weight_decay=1e-4, optimizer="adamw", seed=0, amp=True, device="cuda",
    select_metric="roc_auc", split_mode="cell", val_split=0.2, split_csv=None,
    label_smoothing=0.05, lam_gate=0.3, lam_exp=0.3, lam_lb=0.1,
    lambda_conflict=0.5, coop_eps=1e-8, label_col="cell_type",
    npy_suffix="_polygons_expanded_array.npy", margin=0, cell_type="",
    xenium_label_level="level_2", save_attention=False, attn_maps_per_class=10,
    threshold_strategy="best_f1", threshold=0.5,
)
PATH_KEYS = {"out_dir", "image_dir", "csv_dir", "npy_dir", "prepared_dir",
             "he_tif", "split_csv"}
# Kept as preparation metadata in configs; training reads prepared tables only.
PREPARATION_KEYS = {"align_csv", "cells_csv", "bnd_csv", "annotation_csv",
                    "xenium_mpp", "input_is_micron", "chunksize"}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def load_config(path, overrides=None):
    provided = json.loads(Path(path).read_text())
    unknown = set(provided) - set(DEFAULTS) - PATH_KEYS - PREPARATION_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **provided}
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if cfg["optimizer"] != "adamw":
        raise ValueError("TRACE uses AdamW.")
    if cfg["dataset"] not in ("skin", "xenium"):
        raise ValueError("dataset must be skin or xenium")
    for key in ("batch_size", "epochs", "eval_every", "cell_size", "ctx_size",
                "ctx_dino_input", "e2_attn_tau", "e3_attn_tau", "coop_eps"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["select_metric"] not in ("roc_auc", "pr_auc"):
        raise ValueError("select_metric must be roc_auc or pr_auc")
    if cfg["split_mode"] not in ("cell", "group"):
        raise ValueError("split_mode must be cell or group, or supply split_csv")
    return cfg


def make_dataset(cfg):
    if cfg["dataset"] == "xenium":
        dataset = XeniumDataset(
            prepared_dir=cfg["prepared_dir"], he_tif=cfg.get("he_tif", ""),
            cell_size=cfg["cell_size"], ctx_size=cfg["ctx_size"],
            label_col=canonicalize_xenium_label_col(cfg["xenium_label_level"]),
        )
        if cfg.get("cell_type"):
            labels = np.asarray(dataset.get_labels(), dtype=str)
            match = np.char.lower(np.char.strip(labels)) == cfg["cell_type"].strip().lower()
            if not match.any() or match.all():
                raise ValueError("cell_type must select some, but not all, cells")
            dataset.df.loc[:, dataset.label_col] = np.where(match, "target", "others")
    else:
        dataset = SkinCellsDataset(
            image_dir=cfg["image_dir"], csv_dir=cfg["csv_dir"], npy_dir=cfg["npy_dir"],
            label_col=cfg["label_col"], npy_suffix=cfg["npy_suffix"],
            margin=cfg["margin"], cell_size=cfg["cell_size"], ctx_size=cfg["ctx_size"],
        )
    if not len(dataset):
        raise ValueError("No cells found; check the input paths and annotation match")
    return dataset


def dataset_manifest(dataset):
    if isinstance(dataset, XeniumDataset):
        frame = dataset.df[["cell_id", "poly_idx", dataset.label_col]].copy()
        frame.columns = ["cell_id", "poly_idx", "label"]
        frame.insert(0, "sample", dataset.sample_name)
    else:
        frame = pd.DataFrame(dataset.items)[["sample", "cell_id", "poly_idx", "label"]].copy()
    if frame["label"].isna().any():
        raise ValueError("Missing cell labels; clean annotations before training")
    frame["sample"] = frame["sample"].astype(str)
    frame["cell_id"] = frame["cell_id"].astype(str)
    frame["label"] = frame["label"].astype(str)
    if frame.duplicated(["sample", "cell_id", "poly_idx"]).any():
        raise ValueError("Duplicate sample/cell_id/poly_idx keys")
    return frame.reset_index(drop=True)


def manifest_hash(frame):
    cols = ["sample", "cell_id", "poly_idx", "label"]
    return hashlib.sha256(frame[cols].to_csv(index=False).encode()).hexdigest()


def split_indices(frame, cfg):
    if cfg.get("split_csv"):
        saved = pd.read_csv(cfg["split_csv"], dtype={"sample": str, "cell_id": str})
        keys = ["sample", "cell_id", "poly_idx"]
        joined = frame[keys].merge(saved[keys + ["split"]], on=keys, how="left", validate="one_to_one")
        if len(saved) != len(frame) or not joined["split"].isin(["train", "val"]).all():
            raise ValueError("split_csv must assign every dataset cell exactly once to train or val")
        train_idx = np.flatnonzero(joined["split"].to_numpy() == "train")
        val_idx = np.flatnonzero(joined["split"].to_numpy() == "val")
    elif cfg["split_mode"] == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=cfg["val_split"], random_state=cfg["seed"])
        train_idx, val_idx = next(splitter.split(frame, groups=frame["sample"]))
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(frame)), test_size=cfg["val_split"], random_state=cfg["seed"],
            stratify=frame["label"],
        )
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Both training and validation sets must contain cells")
    if set(frame.iloc[train_idx]["label"]) != set(frame["label"]):
        raise ValueError("Training split is missing a class")
    return np.asarray(train_idx), np.asarray(val_idx)


def build_model(cfg, num_classes, device):
    uni, uni_transform = load_uni_tokens(device=device, model_name=cfg["uni_model_name"])
    dino = load_dino_tokens(device=device, model_name=cfg["dino_model_name"])
    dino_transform, resize_size = build_dino_transform(dino, cfg["ctx_dino_input"])
    kwargs = {k: cfg[k] for k in ("proj_dim", "head_hidden_dim", "head_layers",
              "gate_hidden_dim", "gate_layers", "dropout", "e2_attn_tau", "e3_attn_tau")}
    model = CellExpertModelE1AttnE2E3Attn(
        num_classes=num_classes, uni_dim=get_uni_dim(uni), dino_dim=get_dino_dim(dino),
        uni_model=uni, dino_model=dino, eps=cfg["coop_eps"], **kwargs,
    ).to(device)
    return model, (uni_transform, dino_transform, resize_size)


def make_loader(dataset, indices, classes, cfg, transforms, shuffle=False):
    uni_transform, dino_transform, resize_size = transforms
    label_to_idx = {c: i for i, c in enumerate(classes)}
    collate = partial(build_batch, uni_transform=uni_transform, dino_transform=dino_transform,
                      label_to_idx=label_to_idx, ctx_size=cfg["ctx_size"], ctx_resize_size=resize_size)
    return DataLoader(Subset(dataset, np.asarray(indices).tolist()), batch_size=cfg["batch_size"],
                      shuffle=shuffle, num_workers=cfg["num_workers"], collate_fn=collate,
                      pin_memory=str(cfg["device"]).startswith("cuda"),
                      persistent_workers=cfg["num_workers"] > 0)


def forward_batch(model, batch, device, attention=False):
    return model(batch["cell_img"].to(device), batch["cell_mask"].to(device),
                 batch["ctx_img"].to(device), batch["cell_pos"].to(device), return_attn=attention)


def objective(out, y, cfg, weights):
    ce = partial(ce_smooth, class_weights=weights, label_smoothing=cfg["label_smoothing"])
    def ce_fn(z, labels, reduction):
        return ce(z, labels, reduction=reduction)
    return compute_coop_moe_loss(
        (out["logits_cell"], out["logits_local"], out["logits_ctx"]),
        out["gate_weights"], y, ce_fn, eps=cfg["coop_eps"], lam_gate=cfg["lam_gate"],
        lam_exp=cfg["lam_exp"], lam_lb=cfg["lam_lb"], enable_lb=True,
        enable_conflict_focus=True, lambda_conflict=cfg["lambda_conflict"], conflict_metric="js",
    )


def autocast_context(cfg, device):
    return torch.autocast(device_type=device.type, enabled=bool(cfg["amp"] and device.type == "cuda"))


@torch.no_grad()
def evaluate(model, loader, cfg, device, weights):
    model.eval()
    records = {k: [] for k in ("y", "fused", "e1", "e2", "e3", "alpha")}
    sums, seen = {}, 0
    for batch in tqdm(loader, desc="Validation", leave=False):
        y = batch["labels"].to(device)
        with autocast_context(cfg, device):
            out = forward_batch(model, batch, device)
            loss, stats, details = objective(out, y, cfg, weights)
        if not torch.isfinite(loss) or not torch.isfinite(out["probs"]).all():
            raise FloatingPointError("Non-finite validation loss/probabilities; check inputs and AMP")
        records["y"].append(y.cpu().numpy())
        records["fused"].append(out["probs"].float().cpu().numpy())
        for key, logits_key in (("e1", "logits_cell"), ("e2", "logits_local"), ("e3", "logits_ctx")):
            records[key].append(out[logits_key].softmax(1).float().cpu().numpy())
        records["alpha"].append(out["gate_weights"].float().cpu().numpy())
        for key, val in stats.items():
            sums[key] = sums.get(key, 0.0) + val * len(y)
        seen += len(y)
    return {k: np.concatenate(v) for k, v in records.items()}, {k: v / seen for k, v in sums.items()}


def metrics(records, classes, cfg):
    result = {}
    for key in ("fused", "e1", "e2", "e3"):
        probs = records[key]
        result[key] = dict(compute_auc_metrics(records["y"], probs, np.asarray(classes)),
            classification_report=classification_report(records["y"], probs.argmax(1),
                labels=np.arange(len(classes)), target_names=classes, output_dict=True, zero_division=0),
            confusion_matrix=confusion_matrix(records["y"], probs.argmax(1), labels=np.arange(len(classes))))
        if len(classes) == 2:
            pos_idx = classes.index("Fibroblasts") if "Fibroblasts" in classes else 1
            result[key]["binary_threshold_metrics"] = compute_metrics_binary(
                (records["y"] == pos_idx).astype(int), probs[:, pos_idx],
                cfg["threshold_strategy"], cfg["threshold"])
        else:
            result[key].update(compute_metrics_multiclass(records["y"], probs.argmax(1)))
    return result


def save_evaluation(out_dir, records, stats, classes, frame, cfg):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metrics.json", metrics(records, classes, cfg))
    save_json(out_dir / "coop_moe_stats.json", stats)
    table = frame.copy().reset_index(drop=True)
    table["prediction"] = np.asarray(classes)[records["fused"].argmax(1)]
    for i, cls in enumerate(classes):
        table[f"prob_{cls}"] = records["fused"][:, i]
    for k in range(3):
        table[f"alpha_{k+1}"] = records["alpha"][:, k]
    table.to_csv(out_dir / "predictions.csv", index=False)


def resolve_device(cfg):
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use --device cpu for a CPU run.")
    return device


def fresh_output(path):
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}; choose a new --out_dir")
    out.mkdir(parents=True, exist_ok=True)
    return out


def train(cfg):
    device = resolve_device(cfg)
    set_seed(cfg["seed"])
    dataset = make_dataset(cfg)
    frame = dataset_manifest(dataset)
    classes = sorted(frame["label"].unique().tolist())
    if len(classes) < 2:
        raise ValueError("Classification requires at least two classes")
    train_idx, val_idx = split_indices(frame, cfg)
    out_dir = fresh_output(cfg["out_dir"])
    save_json(out_dir / "config.json", cfg)
    split_frame = frame.copy()
    split_frame["split"] = "train"
    split_frame.loc[val_idx, "split"] = "val"
    split_frame.to_csv(out_dir / "split_manifest.csv", index=False)
    y_train = np.asarray([classes.index(c) for c in frame.iloc[train_idx]["label"]])
    weights = torch.tensor(compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_train),
                           dtype=torch.float32, device=device)
    model, transforms = build_model(cfg, len(classes), device)
    train_loader = make_loader(dataset, train_idx, classes, cfg, transforms, shuffle=True)
    val_loader = make_loader(dataset, val_idx, classes, cfg, transforms)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["amp"] and device.type == "cuda"))
    best_score, best_epoch, history = -float("inf"), None, []
    metric = "pr_auc" if len(classes) == 2 and cfg["select_metric"] == "pr_auc" else "roc_auc"
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}"):
            y = batch["labels"].to(device)
            with autocast_context(cfg, device):
                out = forward_batch(model, batch, device)
                loss, stats, _ = objective(out, y, cfg, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; stop before updating parameters")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += stats["mix_loss"] * len(y)
            seen += len(y)
        row = {"epoch": epoch, "train_mix_loss": loss_sum / seen}
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            records, stats = evaluate(model, val_loader, cfg, device, weights)
            auc = compute_auc_metrics(records["y"], records["fused"], np.asarray(classes))
            score = auc[metric]
            row.update(val_loss=stats["total_loss"], val_roc_auc=auc["roc_auc"], val_pr_auc=auc["pr_auc"],
                       gate_stats=stats)
            if np.isfinite(score) and score > best_score:
                best_score, best_epoch = float(score), epoch
                # Exclude frozen pretrained encoders; reconstruct them from the saved model IDs.
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                         if not k.startswith(("uni.", "dino."))}
                torch.save(dict(model_state_dict=state, config=cfg, classes=classes,
                    class_weights=weights.cpu(), best_epoch=epoch, best_score=best_score,
                    best_metric=metric, train_idx=train_idx.tolist(), val_idx=val_idx.tolist(),
                    dataset_sha256=manifest_hash(frame)), out_dir / "ckpt_best.pt")
        history.append(row)
        save_json(out_dir / "history.json", history)
        print(json.dumps(json_safe(row)))
    if best_epoch is None:
        raise RuntimeError("No finite validation AUC; no best checkpoint selected. Check split class coverage.")
    checkpoint = torch.load(out_dir / "ckpt_best.pt", map_location="cpu", weights_only=True)
    load_head_state(model, checkpoint["model_state_dict"])
    records, stats = evaluate(model, val_loader, cfg, device, weights)
    save_evaluation(out_dir / "diagnostics", records, stats, classes, frame.iloc[val_idx], cfg)
    if cfg["save_attention"]:
        from utils.attention_export import export_attention
        export_attention(model, dataset, val_idx, records, classes, cfg, transforms, device, out_dir / "diagnostics")
    return out_dir / "ckpt_best.pt"


def load_head_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith(("uni.", "dino."))]
    if missing or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def train_main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    for key in ("out_dir", "device"):
        ap.add_argument(f"--{key}")
    for key in ("epochs", "batch_size", "num_workers", "seed"):
        ap.add_argument(f"--{key}", type=int)
    ap.add_argument("--save_attention", action="store_true", default=None)
    args = vars(ap.parse_args())
    train(load_config(args.pop("config"), args))


def evaluate_main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on its saved validation split")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", help="Optional data-path relocation config; model/training parameters stay checkpoint-defined")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device")
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--num_workers", type=int)
    ap.add_argument("--save_attention", action="store_true")
    args = ap.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = dict(checkpoint["config"])
    if args.config:
        relocated = json.loads(Path(args.config).read_text())
        cfg.update({k: v for k, v in relocated.items() if k in PATH_KEYS - {"out_dir", "split_csv"}})
    for key in ("device", "batch_size", "num_workers"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    device = resolve_device(cfg)
    set_seed(cfg["seed"])
    dataset = make_dataset(cfg)
    frame = dataset_manifest(dataset)
    if manifest_hash(frame) != checkpoint["dataset_sha256"]:
        raise ValueError("Dataset IDs, labels, or row order differ from the saved training dataset")
    out_dir = fresh_output(args.out_dir)
    classes, val_idx = checkpoint["classes"], checkpoint["val_idx"]
    model, transforms = build_model(cfg, len(classes), device)
    load_head_state(model, checkpoint["model_state_dict"])
    loader = make_loader(dataset, val_idx, classes, cfg, transforms)
    records, stats = evaluate(model, loader, cfg, device, checkpoint["class_weights"].to(device))
    save_evaluation(out_dir, records, stats, classes, frame.iloc[val_idx], cfg)
    if args.save_attention:
        from utils.attention_export import export_attention
        export_attention(model, dataset, val_idx, records, classes, cfg, transforms, device, out_dir)
