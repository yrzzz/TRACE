"""CPU checks with tiny frozen token encoders; no weights, WSI, or network needed."""
import ast
import copy
import gzip
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import models.tri_expert as released
import trace_runtime as runtime
from losses.coop_moe_loss import compute_coop_moe_loss
from losses.fusion import probability_mix
from models.local_attn_pool import LocalTokenAttentionPool
from utils.attention_export import select_examples, export_attention
from utils.losses import ce_smooth
from xenium_prepare import load_lung_annotations, transform_xy_to_he

torch.set_num_threads(1)


class TinyEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(3, dim)
        self.num_features = dim
        self.num_prefix_tokens = 0
        self.patch_embed = SimpleNamespace(grid_size=(2, 2), patch_size=(16, 16))

    def forward_features(self, image):
        tokens = F.adaptive_avg_pool2d(image, (2, 2)).flatten(2).transpose(1, 2)
        return self.proj(tokens)


def model_kwargs():
    return dict(num_classes=3, uni_dim=8, dino_dim=6, proj_dim=5, head_hidden_dim=7,
                gate_hidden_dim=7, dropout=0.0, uni_model=TinyEncoder(8), dino_model=TinyEncoder(6))


def fake_inputs():
    return (torch.randn(4, 3, 32, 32), torch.ones(4, 32, 32),
            torch.randn(4, 3, 32, 32), torch.ones(4, 2))


def ce(z, y, reduction):
    return F.cross_entropy(z, y, reduction=reduction, label_smoothing=0.05)


class TraceMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_fusion_is_probabilities_not_logits(self):
        z = [torch.tensor([[8., 0.]]), torch.tensor([[0., 2.]]), torch.tensor([[0., 2.]])]
        alpha = torch.tensor([[0.2, 0.4, 0.4]])
        p, logp = probability_mix(z, alpha)
        expected = sum(alpha[:, i:i+1] * zz.softmax(1) for i, zz in enumerate(z))
        torch.testing.assert_close(p, expected)
        self.assertFalse(torch.allclose(p, sum(alpha[:, i:i+1] * zz for i, zz in enumerate(z)).softmax(1)))
        torch.testing.assert_close(logp.exp(), p)

    def test_responsibilities_and_kl(self):
        z = [torch.randn(4, 3, requires_grad=True) for _ in range(3)]
        alpha = torch.randn(4, 3).softmax(1).requires_grad_()
        y = torch.tensor([0, 1, 2, 0])
        loss, stats, detail = compute_coop_moe_loss(z, alpha, y, ce, enable_conflict_focus=False)
        py = torch.stack([zz.softmax(1)[torch.arange(4), y] for zz in z], 1)
        r = alpha * py / ((alpha * py).sum(1, keepdim=True) + 1e-8)
        torch.testing.assert_close(detail["r"], r)
        self.assertFalse(detail["r"].requires_grad)
        kl = (r * ((r + 1e-8).log() - (alpha + 1e-8).log())).sum(1).mean()
        self.assertAlmostEqual(stats["gate_loss"], kl.item(), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(alpha.grad).all())

    def test_attention_weights_are_normalized_and_values_unprojected(self):
        pool = LocalTokenAttentionPool(6, 8, tau=2., cosine=True)
        tokens, query = torch.randn(3, 2, 2, 6), torch.randn(3, 8)
        pooled, attn = pool(tokens, query)
        torch.testing.assert_close(attn.sum((1, 2)), torch.ones(3))
        torch.testing.assert_close(pooled, (tokens * attn[..., None]).sum((1, 2)))
        self.assertNotIn("k_proj.weight", pool.state_dict())

    def test_model_loss_inference_and_backward(self):
        model = released.CellExpertModelE1AttnE2E3Attn(**model_kwargs())
        out = model(*fake_inputs(), return_attn=True)
        cfg = runtime.DEFAULTS
        loss, _, detail = runtime.objective(out, torch.tensor([0, 1, 2, 0]), cfg, torch.ones(3))
        torch.testing.assert_close(out["probs"], detail["p_mix"], rtol=0, atol=0)
        torch.testing.assert_close(out["logits"], detail["z_mix"], rtol=0, atol=0)
        self.assertEqual(tuple(out["gate_weights"].shape), (4, 3))
        loss.backward()
        for name, param in model.named_parameters():
            if name.startswith(("uni.", "dino.")):
                self.assertIsNone(param.grad)
            else:
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)

    @unittest.skipUnless(os.environ.get("TRACE_REFERENCE_ROOT"), "Set TRACE_REFERENCE_ROOT for source regression")
    def test_research_source_regression(self):
        source = Path(os.environ["TRACE_REFERENCE_ROOT"]) / "models/model.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CellExpertModelE1AttnE2E3Attn")
        namespace = dict(vars(released))
        namespace["global_pool"] = lambda x: x.mean((1, 2))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
        kwargs = model_kwargs()
        old = namespace[cls.name](**copy.deepcopy(kwargs), e2_attn_query="e1", ctx_attn_query="e1",
                                  e2_attn_cosine=True, e3_attn_cosine=True, e2_attn_tau=2.)
        new = released.CellExpertModelE1AttnE2E3Attn(**copy.deepcopy(kwargs))
        new.load_state_dict(old.state_dict(), strict=True)
        old.eval()
        new.eval()
        inputs = fake_inputs()
        old_out, new_out = old(*inputs, return_attn=True), new(*inputs, return_attn=True)
        for key in new_out.keys() - {"logits", "probs"}:
            torch.testing.assert_close(old_out[key], new_out[key], rtol=0, atol=0)
        # Independently load the original loss function, rather than comparing it to itself.
        loss_path = Path(os.environ["TRACE_REFERENCE_ROOT"]) / "losses/coop_moe_loss.py"
        loss_namespace = {}
        exec(compile(loss_path.read_text(), str(loss_path), "exec"), loss_namespace)
        y = torch.tensor([0, 1, 2, 0])
        old_loss, _, detail = loss_namespace["compute_coop_moe_loss"](
            [old_out[k] for k in ("logits_cell", "logits_local", "logits_ctx")],
            old_out["gate_weights"], y, ce, lam_gate=.3, lam_exp=.3, lam_lb=.1,
            enable_conflict_focus=True)
        new_loss, _, _ = runtime.objective(new_out, y, runtime.DEFAULTS, torch.ones(3))
        torch.testing.assert_close(new_out["probs"], detail["p_mix"], rtol=0, atol=0)
        torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
        old_loss.backward()
        new_loss.backward()
        for (name, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)


class DataTests(unittest.TestCase):
    def test_annotation_gzip_and_level_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotation.csv.gz"
            with gzip.open(path, "wt") as file:
                file.write("Unnamed: 0,level1,level2,level3\na-1,A,B,C\n")
            frame = load_lung_annotations(path)
            self.assertEqual(frame.iloc[0]["join_id"], "a-1")
            self.assertEqual(frame.iloc[0]["level_3"], "C")

    def test_alignment_units(self):
        u, v = transform_xy_to_he(np.array([2.]), np.array([4.]), np.eye(3), 0.5, True)
        np.testing.assert_array_equal(u, [4.])
        np.testing.assert_array_equal(v, [8.])

    def test_pair_selection_all_outcomes(self):
        y = np.array([0, 0, 1, 1])
        selected = select_examples(y, np.array([0, 1, 1, 0]), np.array([.7, .9, .8, .6]), 2, 1)
        self.assertEqual(selected, [0, 1, 2, 3])

    def test_explicit_split_and_fingerprint(self):
        frame = pd.DataFrame(dict(sample=["a"] * 4, cell_id=list("abcd"), poly_idx=range(4), label=["X", "Y"] * 2))
        with tempfile.TemporaryDirectory() as tmp:
            saved = frame.assign(split=["train", "train", "val", "val"])
            path = Path(tmp) / "split.csv"
            saved.iloc[::-1].to_csv(path, index=False)
            tr, val = runtime.split_indices(frame, {"split_csv": str(path)})
            np.testing.assert_array_equal(tr, [0, 1])
            np.testing.assert_array_equal(val, [2, 3])
            self.assertNotEqual(runtime.manifest_hash(frame), runtime.manifest_hash(frame.iloc[::-1]))

    def test_all_published_configs_parse(self):
        for path in (Path(__file__).parents[1] / "configs").glob("*.json"):
            runtime.load_config(path)


class PipelineTests(unittest.TestCase):
    def test_train_reload_evaluate_and_paired_maps(self):
        import cv2
        import tifffile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("images", "labels", "polygons"):
                (root / folder).mkdir()
            image = np.random.default_rng(3).integers(0, 255, (64, 64, 3), dtype=np.uint8)
            tifffile.imwrite(root / "images/slide.tif", image)
            pd.DataFrame(dict(cell_id=[f"cell{i}" for i in range(12)], poly_idx=range(12),
                              cell_type=["A", "B"] * 6)).to_csv(root / "labels/slide.csv", index=False)
            polygons = np.empty(12, dtype=object)
            for i in range(12):
                polygons[i] = np.array([[20, 20, 40, 40], [20, 40, 40, 20]], dtype=np.float32)
            np.save(root / "polygons/slide_polygons_expanded_array.npy", polygons)
            cfg = {**runtime.DEFAULTS, "dataset": "skin", "image_dir": str(root / "images"),
                   "csv_dir": str(root / "labels"), "npy_dir": str(root / "polygons"),
                   "out_dir": str(root / "run"), "epochs": 1, "batch_size": 4, "num_workers": 0,
                   "device": "cpu", "amp": False, "cell_size": 32, "ctx_size": 48, "val_split": .5,
                   "save_attention": True, "attn_maps_per_class": 1}
            def fake_build(config, nclasses, device):
                kwargs = model_kwargs()
                kwargs["num_classes"] = nclasses
                model = released.CellExpertModelE1AttnE2E3Attn(**kwargs)
                transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
                return model, (transform, transform, 224)
            with patch.object(runtime, "build_model", fake_build):
                checkpoint_path = runtime.train(cfg)
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                self.assertFalse(any(k.startswith(("uni.", "dino.")) for k in checkpoint["model_state_dict"]))
                dataset = runtime.make_dataset(cfg)
                model, transform = fake_build(cfg, 2, "cpu")
                runtime.load_head_state(model, checkpoint["model_state_dict"])
                self.assertTrue((root / "run/diagnostics/metrics.json").exists())
                for prefix in ("local", "ctx"):
                    summary = json.loads((root / f"run/diagnostics/{prefix}_attn_maps/summary.json").read_text())
                    self.assertTrue(summary["saved_sample_keys"])
                local = json.loads((root / "run/diagnostics/local_attn_maps/summary.json").read_text())
                context = json.loads((root / "run/diagnostics/ctx_attn_maps/summary.json").read_text())
                self.assertEqual(local["saved_sample_keys"], context["saved_sample_keys"])
                # Exercise the actual CLI reload path with deterministic fake pretrained weights.
                with patch("sys.argv", ["evaluate.py", "--checkpoint", str(checkpoint_path),
                                       "--out_dir", str(root / "eval"), "--device", "cpu"]):
                    runtime.evaluate_main()
                self.assertTrue((root / "eval/predictions.csv").exists())
                before = pd.read_csv(root / "run/diagnostics/predictions.csv")
                after = pd.read_csv(root / "eval/predictions.csv")
                np.testing.assert_array_equal(before.filter(regex="^prob_").to_numpy(),
                                              after.filter(regex="^prob_").to_numpy())


if __name__ == "__main__":
    unittest.main()
