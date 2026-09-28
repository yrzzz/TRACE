"""Export paired local/context pooling weights for correct and incorrect cells."""
import re
from pathlib import Path

import cv2
import numpy as np
import torch


def select_examples(y, pred, confidence, num_classes, per_class):
    selected = []
    for cls in range(num_classes):
        for correct in (True, False):
            candidates = np.flatnonzero((y == cls) & ((pred == y) == correct))
            selected.extend(candidates[np.argsort(-confidence[candidates])][:per_class].tolist())
    return selected


def safe_name(value):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))


def save_view(directory, patch, weights, prefix, mask=None):
    directory.mkdir(parents=True, exist_ok=True)
    height, width = patch.shape[:2]
    heat = cv2.resize(weights, (width, height), interpolation=cv2.INTER_LINEAR)
    heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-8)
    heat_u8 = (heat * 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = (0.55 * patch.astype(float) + 0.45 * color).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(directory / ("cell_patch.png" if prefix == "local" else "ctx_patch.png")),
                cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(directory / f"{prefix}_attn_heatmap.png"), heat_u8)
    cv2.imwrite(str(directory / f"{prefix}_attn_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    # PNGs are contrast-scaled for display; this array retains the true normalized weights.
    np.save(directory / "attention_weights.npy", weights)
    if mask is not None:
        cv2.imwrite(str(directory / "mask.png"), (mask * 255).astype(np.uint8))
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = patch.copy()
        cv2.drawContours(polygon, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(str(directory / "polygon_overlay.png"), cv2.cvtColor(polygon, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def export_attention(model, dataset, val_idx, records, classes, cfg, transforms, device, out_dir):
    from trace_runtime import make_loader, forward_batch, autocast_context, save_json
    model.eval()
    selected = select_examples(records["y"], records["fused"].argmax(1),
                               records["fused"].max(1), len(classes), cfg["attn_maps_per_class"])
    summary = {"query_source": "e1", "selection": "top confidence, correct and wrong per true class",
               "saved_sample_keys": [], "counts_per_class_outcome": {c: {"correct": 0, "wrong": 0} for c in classes}}
    for i in selected:
        index = int(val_idx[i])
        loader = make_loader(dataset, [index], classes, {**cfg, "num_workers": 0}, transforms)
        batch = next(iter(loader))
        with autocast_context(cfg, device):
            out = forward_batch(model, batch, device, attention=True)
        cell_patch, mask, context_patch, _, _, meta = dataset[index]
        cls = classes[int(records["y"][i])]
        pred = int(records["fused"][i].argmax())
        correct = pred == records["y"][i]
        key = safe_name(f"{meta.sample}__{meta.cell_id}__{meta.poly_idx}")
        payload = dict(sample=meta.sample, cell_id=str(meta.cell_id), poly_idx=int(meta.poly_idx),
                       true_label=cls, pred_label=classes[pred], correct=bool(correct),
                       pred_prob=float(records["fused"][i, pred]), query_source="e1",
                       gate_weights=records["alpha"][i].tolist())
        for prefix, patch in (("local", cv2.resize(cell_patch, (224, 224), interpolation=cv2.INTER_CUBIC)),
                              ("ctx", context_patch)):
            weights = out[f"{prefix}_attn_map"][0].float().cpu().numpy()
            directory = Path(out_dir) / f"{prefix}_attn_maps" / safe_name(cls) / key
            mask224 = cv2.resize(mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
            save_view(directory, patch, weights, prefix, mask224 if prefix == "local" else None)
            save_json(directory / "meta.json", {**payload, "attn_entropy": float(-(weights * np.log(weights + 1e-8)).sum())})
        summary["saved_sample_keys"].append(key)
        summary["counts_per_class_outcome"][cls]["correct" if correct else "wrong"] += 1
    for prefix in ("local", "ctx"):
        directory = Path(out_dir) / f"{prefix}_attn_maps"
        directory.mkdir(parents=True, exist_ok=True)
        save_json(directory / "summary.json", summary)
