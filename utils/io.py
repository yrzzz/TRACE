#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Optional, List, Any

import cv2
import numpy as np
from PIL import Image
import torch


def apply_image_transform(image_rgb: np.ndarray, transform):
    if not isinstance(image_rgb, Image.Image):
        image_rgb = Image.fromarray(image_rgb)
    return transform(image_rgb)


def resize_mask(mask: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized


def build_batch(
    samples: List[Any],
    uni_transform,
    dino_transform,
    label_to_idx: Optional[dict] = None,
    ctx_size: Optional[int] = None,
    ctx_resize_size: Optional[int] = None,
    use_ctx: bool = True,
    return_meta: bool = True,
):
    cell_imgs = []
    cell_masks = []
    ctx_imgs = []
    cell_pos = []
    labels = []
    metas = []

    for cell_patch, cell_mask, ctx_patch, cell_pos_in_ctx, label, meta in samples:
        cell_img = apply_image_transform(cell_patch, uni_transform)
        cell_mask_224 = resize_mask(cell_mask, (224, 224))
        cell_mask_t = torch.from_numpy(cell_mask_224).float()
        if use_ctx:
            ctx_img = apply_image_transform(ctx_patch, dino_transform)
        else:
            ctx_img = None

        target_size = ctx_resize_size if ctx_resize_size is not None else ctx_size
        if use_ctx and target_size is not None:
            h, w = ctx_patch.shape[:2]
            if (h, w) != (target_size, target_size):
                sx = float(target_size) / float(w)
                sy = float(target_size) / float(h)
                cell_pos_scaled = (cell_pos_in_ctx[0] * sx, cell_pos_in_ctx[1] * sy)
            else:
                cell_pos_scaled = cell_pos_in_ctx
        elif use_ctx:
            cell_pos_scaled = cell_pos_in_ctx
        else:
            cell_pos_scaled = (0.0, 0.0)

        cell_imgs.append(cell_img)
        cell_masks.append(cell_mask_t)
        if use_ctx:
            ctx_imgs.append(ctx_img)
            cell_pos.append(torch.tensor(cell_pos_scaled, dtype=torch.float32))

        if label_to_idx is None:
            labels.append(label)
        else:
            labels.append(label_to_idx[label])
        if return_meta:
            metas.append(meta)

    batch = {
        "cell_img": torch.stack(cell_imgs, dim=0),
        "cell_mask": torch.stack(cell_masks, dim=0),
        "ctx_img": torch.stack(ctx_imgs, dim=0) if use_ctx else None,
        "cell_pos": torch.stack(cell_pos, dim=0) if use_ctx else None,
        "labels": labels if label_to_idx is None else torch.tensor(labels, dtype=torch.long),
    }
    if return_meta:
        batch["meta"] = metas
    return batch
