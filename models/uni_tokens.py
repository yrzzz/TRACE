#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Tuple

import timm
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.models import hub
from timm.models.vision_transformer import VisionTransformer
from timm.layers.mlp import GluMlp, Mlp


def _get_default_norm_cfg(model):
    cfg = getattr(model, "default_cfg", {}) or {}
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    return mean, std


def build_uni_transform(model, mode: str = "simple"):
    if mode == "timm":
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        cfg = resolve_data_config(getattr(model, "pretrained_cfg", {}) or {}, model=model)
        cfg["input_size"] = (3, 224, 224)
        return create_transform(**cfg)

    mean, std = _get_default_norm_cfg(model)
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _legacy_uni_kwargs():
    mlp_layer = getattr(timm.layers, "SwiGLUPacked", GluMlp)
    if mlp_layer is GluMlp:
        print("[WARN] timm.layers.SwiGLUPacked not available; using GluMlp fallback.")
    return {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": mlp_layer,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }


def _infer_uni_config_from_state_dict(state_dict):
    embed_dim = state_dict["pos_embed"].shape[-1]
    patch_w = state_dict["patch_embed.proj.weight"].shape[-1]
    depth = len({k.split(".")[1] for k in state_dict.keys() if k.startswith("blocks.")})
    fc1_out = state_dict["blocks.0.mlp.fc1.weight"].shape[0]
    fc2_in = state_dict["blocks.0.mlp.fc2.weight"].shape[1]
    gated_mlp = fc1_out == 2 * fc2_in
    mlp_ratio = float(fc1_out) / float(embed_dim) if gated_mlp else float(fc2_in) / float(embed_dim)
    has_cls = "cls_token" in state_dict
    reg_tokens = int(state_dict["reg_token"].shape[1]) if "reg_token" in state_dict else 0
    use_layer_scale = any(k.endswith("ls1.gamma") for k in state_dict.keys())
    num_patches = state_dict["pos_embed"].shape[1]
    return {
        "embed_dim": embed_dim,
        "patch_size": patch_w,
        "depth": depth,
        "mlp_ratio": mlp_ratio,
        "gated_mlp": gated_mlp,
        "has_cls": has_cls,
        "reg_tokens": reg_tokens,
        "use_layer_scale": use_layer_scale,
        "num_patches": num_patches,
    }


def _infer_num_heads(embed_dim: int) -> int:
    for head_dim in (64, 96, 48, 32):
        if embed_dim % head_dim == 0:
            return embed_dim // head_dim
    raise ValueError(f"Cannot infer num_heads for embed_dim={embed_dim}")


def _build_uni_from_hf(model_name: str):
    hf_id = model_name.replace("hf-hub:", "")
    state_dict = hub.load_state_dict_from_hf(hf_id)
    cfg = _infer_uni_config_from_state_dict(state_dict)
    num_heads = _infer_num_heads(cfg["embed_dim"])

    # UNI2-h uses pos_embed without class token; honor that layout.
    img_size = 224
    num_patches = (img_size // cfg["patch_size"]) ** 2
    no_embed_class = cfg["has_cls"] and cfg["num_patches"] == num_patches

    mlp_layer = GluMlp if cfg["gated_mlp"] else Mlp
    model = VisionTransformer(
        img_size=img_size,
        patch_size=cfg["patch_size"],
        in_chans=3,
        num_classes=0,
        global_pool="token",
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=num_heads,
        mlp_ratio=cfg["mlp_ratio"],
        mlp_layer=mlp_layer,
        qkv_bias=True,
        class_token=cfg["has_cls"],
        no_embed_class=no_embed_class,
        reg_tokens=cfg["reg_tokens"],
        init_values=1e-5 if cfg["use_layer_scale"] else None,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] UNI state_dict load: missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def _build_legacy_from_state_dict(model_name: str):
    hf_id = model_name.replace("hf-hub:", "")
    state_dict = hub.load_state_dict_from_hf(hf_id)
    model = timm.create_model("vit_giant_patch14_224", pretrained=False, **_legacy_uni_kwargs())
    model.load_state_dict(state_dict, strict=True)
    return model


def load_uni_tokens(
    device: str,
    model_name: str = "hf-hub:MahmoodLab/UNI2-h",
    loader: str = "auto",
    transform_mode: str = "simple",
):
    loader = loader.lower()
    if loader == "legacy_timm":
        try:
            model = timm.create_model(model_name, pretrained=True, **_legacy_uni_kwargs())
        except Exception as e:
            print(f"[WARN] legacy timm load failed: {e}")
            print("[WARN] Falling back to legacy state_dict load.")
            model = _build_legacy_from_state_dict(model_name)
    elif loader == "legacy_state_dict":
        model = _build_legacy_from_state_dict(model_name)
    elif loader == "auto":
        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=0)
        except Exception as e:
            print(f"[WARN] timm pretrained load failed: {e}")
            print("[WARN] Falling back to custom UNI loader from HF state_dict.")
            model = _build_uni_from_hf(model_name)
    else:
        raise ValueError(f"Unknown UNI loader: {loader}")

    model.eval()
    model.to(device)
    transform = build_uni_transform(model, mode=transform_mode)
    return model, transform


def _extract_tokens(output):
    if isinstance(output, dict):
        for key in ("x", "last_hidden_state", "tokens", "patch_tokens"):
            if key in output:
                output = output[key]
                break
    if isinstance(output, (list, tuple)):
        output = output[0]
    if output.dim() == 4:
        # BCHW -> BHW D
        output = output.flatten(2).transpose(1, 2)
    if output.dim() != 3:
        raise ValueError(f"Unexpected token output shape: {tuple(output.shape)}")
    return output


def _num_prefix_tokens(model) -> int:
    if hasattr(model, "num_prefix_tokens"):
        return int(model.num_prefix_tokens)
    if hasattr(model, "cls_token"):
        return 1
    return 0


def _infer_grid_size(model, num_patches: int) -> Tuple[int, int]:
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
        gh, gw = model.patch_embed.grid_size
        if gh * gw == num_patches:
            return int(gh), int(gw)
    g = int(math.sqrt(num_patches))
    if g * g != num_patches:
        raise ValueError(f"Cannot infer grid size from num_patches={num_patches}")
    return g, g


def uni_forward_tokens(model, x: torch.Tensor) -> torch.Tensor:
    tokens = _extract_tokens(model.forward_features(x))
    n_prefix = _num_prefix_tokens(model)
    patch_tokens = tokens[:, n_prefix:, :]
    bsz, num_patches, dim = patch_tokens.shape
    gh, gw = _infer_grid_size(model, num_patches)
    tokens_grid = patch_tokens.reshape(bsz, gh, gw, dim)
    return tokens_grid


def uni_forward_tokens_all(model, x: torch.Tensor) -> torch.Tensor:
    return _extract_tokens(model.forward_features(x))


def get_uni_embedding(
    model: torch.nn.Module,
    x: torch.Tensor,
    pool_mode: str = "patch_mean",
    proj_layer: torch.nn.Module = None,
    return_tokens: bool = False,
):
    if pool_mode == "timm_global":
        emb = model(x)
        if emb.dim() != 2:
            raise ValueError(f"timm_global expects [B,D], got {tuple(emb.shape)}")
        return (emb, None) if return_tokens else emb

    tokens = uni_forward_tokens_all(model, x)
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens [B,N,D], got {tuple(tokens.shape)}")

    n_prefix = _num_prefix_tokens(model)
    if pool_mode == "patch_mean":
        patch_tokens = tokens[:, n_prefix:, :]
        emb = patch_tokens.mean(dim=1)
    elif pool_mode == "all_tokens_mean":
        emb = tokens.mean(dim=1)
    elif pool_mode == "cls_only":
        if n_prefix < 1:
            raise ValueError("cls_only requested but model has no prefix tokens")
        emb = tokens[:, 0, :]
    elif pool_mode == "cls_plus_patchmean":
        if n_prefix < 1:
            raise ValueError("cls_plus_patchmean requested but model has no prefix tokens")
        if proj_layer is None:
            raise ValueError("cls_plus_patchmean requires proj_layer (Linear) to project 2D->D")
        patch_tokens = tokens[:, n_prefix:, :]
        patch_mean = patch_tokens.mean(dim=1)
        cls = tokens[:, 0, :]
        emb = proj_layer(torch.cat([cls, patch_mean], dim=1))
    else:
        raise ValueError(f"Unknown pool_mode: {pool_mode}")

    return (emb, tokens) if return_tokens else emb


def get_uni_dim(model) -> int:
    if hasattr(model, "num_features"):
        return int(model.num_features)
    if hasattr(model, "embed_dim"):
        return int(model.embed_dim)
    raise ValueError("Cannot infer UNI embedding dim")
