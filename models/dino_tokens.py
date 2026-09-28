#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Tuple

import timm
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def _get_default_norm_cfg(model):
    cfg = getattr(model, "default_cfg", {}) or {}
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    return mean, std


def _align_ctx_size(ctx_size: int, patch_size: int, mode: str = "nearest") -> int:
    if patch_size <= 0:
        return int(ctx_size)
    ratio = float(ctx_size) / float(patch_size)
    if mode == "floor":
        k = max(1, int(math.floor(ratio)))
    elif mode == "ceil":
        k = max(1, int(math.ceil(ratio)))
    else:
        k = max(1, int(round(ratio)))
    return int(k * patch_size)


def build_dino_transform(
    model,
    ctx_size: int,
    align_to_patch: bool = True,
    align_mode: str = "nearest",
):
    mean, std = _get_default_norm_cfg(model)
    patch_size = _get_patch_size(model)
    resize_size = int(ctx_size)
    if align_to_patch:
        resize_size = _align_ctx_size(ctx_size, patch_size, mode=align_mode)
        if resize_size != ctx_size:
            print(
                f"[WARN] ctx_size={ctx_size} not divisible by patch_size={patch_size}; "
                f"resizing to {resize_size} for DINO."
            )
    transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transform, resize_size


def load_dino_tokens(
    device: str,
    model_name: str = "vit_base_patch16_dinov3.lvd1689m",
    dynamic_img_size: bool = True,
):
    try:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=dynamic_img_size,
        )
    except TypeError:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    return model


def _extract_tokens(output):
    if isinstance(output, dict):
        for key in ("x", "last_hidden_state", "tokens", "patch_tokens"):
            if key in output:
                output = output[key]
                break
    if isinstance(output, (list, tuple)):
        output = output[0]
    if output.dim() == 4:
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


def _get_patch_size(model) -> int:
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
        ps = model.patch_embed.patch_size
        if isinstance(ps, (tuple, list)):
            return int(ps[0])
        return int(ps)
    raise ValueError("Cannot infer patch size")


def dino_forward_tokens(model, x: torch.Tensor):
    tokens = _extract_tokens(model.forward_features(x))
    n_prefix = _num_prefix_tokens(model)
    patch_tokens = tokens[:, n_prefix:, :]
    bsz, num_patches, dim = patch_tokens.shape
    gh, gw = _infer_grid_size(model, num_patches)
    tokens_grid = patch_tokens.reshape(bsz, gh, gw, dim)
    patch_size = _get_patch_size(model)
    return tokens_grid, patch_size


def get_dino_dim(model) -> int:
    if hasattr(model, "num_features"):
        return int(model.num_features)
    if hasattr(model, "embed_dim"):
        return int(model.embed_dim)
    raise ValueError("Cannot infer DINO embedding dim")
