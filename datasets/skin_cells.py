#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset


# ---------- 1) label grouping ----------
def add_fibroblast_group(
    df: pd.DataFrame,
    cell_type_col: str = "cell_type",
    new_col: str = "group",
) -> pd.DataFrame:
    out = df.copy()
    ct = out[cell_type_col].astype("string").str.strip().str.lower()
    out[new_col] = ct.eq("fibroblasts").map({True: "Fibroblasts", False: "others"})
    return out


def add_selected_group(
    df: pd.DataFrame,
    target_cell_types: List[str],
    cell_type_col: str = "cell_type",
    new_col: str = "__target_group__",
    pos_label: str = "target",
    neg_label: str = "others",
) -> pd.DataFrame:
    out = df.copy()
    ct = out[cell_type_col].astype("string").str.strip().str.lower()
    targets = {t.strip().lower() for t in target_cell_types if str(t).strip()}
    if not targets:
        raise ValueError("target_cell_types is empty")
    out[new_col] = ct.isin(targets).map({True: pos_label, False: neg_label})
    return out


# ---------- 2) pairing ----------
def list_image_csv_npy_triples(
    image_dir: str,
    csv_dir: str,
    npy_dir: str,
    npy_suffix: str = "_polygons_expanded_array.npy",
) -> List[Tuple[str, str, str, str]]:
    img_paths = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        img_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    img_paths = sorted(img_paths)

    triples = []
    missing = []
    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        csv_path = os.path.join(csv_dir, f"{stem}.csv")
        npy_path = os.path.join(npy_dir, f"{stem}{npy_suffix}")

        if not os.path.exists(csv_path) or not os.path.exists(npy_path):
            missing.append(stem)
            continue

        triples.append((stem, img_path, csv_path, npy_path))

    if missing:
        print(f"[WARN] missing CSV/NPY for {len(missing)} images. Example:", missing[:5])
    print(f"[INFO] paired samples: {len(triples)}")
    return triples


# ---------- 3) polygon utils ----------
def _parse_poly_indices(df: pd.DataFrame) -> np.ndarray:
    if "poly_idx" in df.columns:
        return df["poly_idx"].astype(int).to_numpy()
    if "index" in df.columns:
        idx = df["index"].astype(str).str.extract(r"(\d+)$")[0]
        if idx.isna().any():
            bad = df.loc[idx.isna(), "index"].head(5).tolist()
            raise ValueError(f"Failed parsing poly_idx from index for rows: {bad}")
        return idx.astype(int).to_numpy()
    raise ValueError("CSV must contain 'poly_idx' or 'index' to map polygons")


def polygon_bbox_center(polygon: np.ndarray) -> Tuple[float, float, float, float]:
    ys = polygon[0, :]
    xs = polygon[1, :]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    bw = x_max - x_min
    bh = y_max - y_min
    return cx, cy, bw, bh


# ---------- 4) cropping ----------
def crop_square_patch(
    image_rgb: np.ndarray,
    cx: float,
    cy: float,
    side: int,
    pad_mode: str = "edge",
    constant_value: int = 255,
) -> Tuple[np.ndarray, int, int]:
    H, W = image_rgb.shape[:2]
    side = int(np.ceil(side))
    side = max(side, 2)

    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))

    # If side exceeds image, match original behavior: crop largest square, then pad.
    if side > W or side > H:
        s2 = min(W, H)
        x0_c = max(0, min(W - s2, x0))
        y0_c = max(0, min(H - s2, y0))
        crop = image_rgb[y0_c:y0_c + s2, x0_c:x0_c + s2, :].astype(np.uint8)

        pad = side - s2
        pt = pad // 2
        pb = pad - pt
        pl = pad // 2
        pr = pad - pl

        if pad_mode == "edge":
            crop = np.pad(crop, ((pt, pb), (pl, pr), (0, 0)), mode="edge")
        else:
            crop = np.pad(
                crop,
                ((pt, pb), (pl, pr), (0, 0)),
                mode="constant",
                constant_values=constant_value,
            )

        # virtual top-left for alignment with padded patch
        x0_v = x0_c - pl
        y0_v = y0_c - pt
        return crop, x0_v, y0_v

    # Normal case: clamp to keep patch inside image
    x0_c = max(0, min(W - side, x0))
    y0_c = max(0, min(H - side, y0))
    crop = image_rgb[y0_c:y0_c + side, x0_c:x0_c + side, :].astype(np.uint8)
    return crop, x0_c, y0_c


def crop_cell_patch_from_polygon_bbox(
    image_rgb: np.ndarray,
    polygon: np.ndarray,
    margin: int = 0,
    pad_mode: str = "edge",
) -> Tuple[np.ndarray, int, int, int]:
    cx, cy, bw, bh = polygon_bbox_center(polygon)
    side = int(np.ceil(max(bw, bh) + 2 * margin))
    side = max(side, 2)
    patch, x0, y0 = crop_square_patch(
        image_rgb=image_rgb,
        cx=cx,
        cy=cy,
        side=side,
        pad_mode=pad_mode,
    )
    return patch, x0, y0, side


def crop_context_patch(
    image_rgb: np.ndarray,
    cx: float,
    cy: float,
    ctx_size: int,
    pad_mode: str = "edge",
) -> Tuple[np.ndarray, int, int, int]:
    patch, x0, y0 = crop_square_patch(
        image_rgb=image_rgb,
        cx=cx,
        cy=cy,
        side=ctx_size,
        pad_mode=pad_mode,
    )
    return patch, x0, y0, int(ctx_size)


def rasterize_polygon_mask(
    polygon: np.ndarray,
    x0: int,
    y0: int,
    side: int,
) -> np.ndarray:
    xs = polygon[1, :]
    ys = polygon[0, :]
    pts = np.stack([xs - x0, ys - y0], axis=1)
    pts = np.round(pts).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))

    mask = np.zeros((side, side), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


@dataclass
class CellMeta:
    sample: str
    cell_id: Any
    poly_idx: int


class SkinCellsDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        csv_dir: str,
        npy_dir: str,
        label_col: str = "group",
        add_group: bool = True,
        group_source_col: str = "cell_type",
        npy_suffix: str = "_polygons_expanded_array.npy",
        margin: int = 0,
        cell_size: int = 0,
        ctx_size: int = 1024,
        skip_ctx: bool = False,
        cache_images: bool = True,
        cache_polygons: bool = True,
    ) -> None:
        self.triples = list_image_csv_npy_triples(image_dir, csv_dir, npy_dir, npy_suffix)
        self.margin = int(margin)
        self.cell_size = int(cell_size) if cell_size else 0
        self.ctx_size = int(ctx_size)
        self.skip_ctx = bool(skip_ctx)
        self.cache_images = cache_images
        self.cache_polygons = cache_polygons

        self.items: List[Dict[str, Any]] = []
        requested_targets = [x.strip() for x in str(label_col).split(",") if x.strip()]
        target_group_logged = False

        for sample, img_path, csv_path, npy_path in self.triples:
            df = pd.read_csv(csv_path)
            if add_group and group_source_col in df.columns:
                df = add_fibroblast_group(df, cell_type_col=group_source_col, new_col="group")

            label_col_resolved = label_col
            if label_col_resolved not in df.columns:
                if requested_targets and group_source_col in df.columns:
                    df = add_selected_group(
                        df,
                        target_cell_types=requested_targets,
                        cell_type_col=group_source_col,
                        new_col="__target_group__",
                        pos_label="target",
                        neg_label="others",
                    )
                    label_col_resolved = "__target_group__"
                    if not target_group_logged:
                        label_target_str = ", ".join(requested_targets)
                        print(
                            f"[INFO] label_col interpreted as target cell types: [{label_target_str}] "
                            f"from '{group_source_col}' -> target vs others"
                        )
                        target_group_logged = True
                else:
                    raise ValueError(f"CSV missing label_col='{label_col}' | file={csv_path}")

            poly_indices = _parse_poly_indices(df)
            labels = df[label_col_resolved].to_numpy()
            cell_ids = df["cell_id"].to_numpy() if "cell_id" in df.columns else df.index.to_numpy()

            for i in range(len(df)):
                self.items.append(
                    {
                        "sample": sample,
                        "img_path": img_path,
                        "npy_path": npy_path,
                        "poly_idx": int(poly_indices[i]),
                        "label": labels[i],
                        "cell_id": cell_ids[i],
                    }
                )

        print(f"[INFO] total cells indexed: {len(self.items)}")

        self._img_cache: Dict[str, np.ndarray] = {}
        self._poly_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, img_path: str) -> np.ndarray:
        if self.cache_images and img_path in self._img_cache:
            return self._img_cache[img_path]
        image = io.imread(img_path)
        if image.ndim == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image
        if self.cache_images:
            self._img_cache[img_path] = image_rgb
        return image_rgb

    def _load_polygons(self, npy_path: str) -> np.ndarray:
        if self.cache_polygons and npy_path in self._poly_cache:
            return self._poly_cache[npy_path]
        polygons = np.load(npy_path, allow_pickle=True)
        if self.cache_polygons:
            self._poly_cache[npy_path] = polygons
        return polygons

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image_rgb = self._load_image(item["img_path"])
        polygons = self._load_polygons(item["npy_path"])

        poly_idx = item["poly_idx"]
        if poly_idx < 0 or poly_idx >= len(polygons):
            raise IndexError(
                f"poly_idx out of range: {poly_idx} | npy={item['npy_path']} | n_polygons={len(polygons)}"
            )
        polygon = np.asarray(polygons[poly_idx])
        if polygon.shape[0] != 2:
            raise ValueError(f"Polygon must have shape (2,P). Got {polygon.shape} for idx={poly_idx}")

        # cell patch + mask (polygon bbox square)
        if self.cell_size > 0:
            cx, cy, _, _ = polygon_bbox_center(polygon)
            cell_patch, x0, y0 = crop_square_patch(
                image_rgb=image_rgb,
                cx=cx,
                cy=cy,
                side=self.cell_size,
            )
            side = int(self.cell_size)
        else:
            cell_patch, x0, y0, side = crop_cell_patch_from_polygon_bbox(
                image_rgb=image_rgb,
                polygon=polygon,
                margin=self.margin,
            )
        cell_mask = rasterize_polygon_mask(polygon, x0, y0, side)

        # context patch (optional; skipped for models that never consume context, e.g., hover).
        cx, cy, _, _ = polygon_bbox_center(polygon)
        if self.skip_ctx:
            ctx_patch = np.zeros((1, 1, 3), dtype=np.uint8)
            cell_pos_in_ctx = (0.0, 0.0)
        else:
            ctx_patch, ctx_x0, ctx_y0, _ctx_size = crop_context_patch(
                image_rgb=image_rgb,
                cx=cx,
                cy=cy,
                ctx_size=self.ctx_size,
            )
            cell_pos_in_ctx = (float(cx - ctx_x0), float(cy - ctx_y0))

        meta = CellMeta(sample=item["sample"], cell_id=item["cell_id"], poly_idx=poly_idx)

        return (
            cell_patch,
            cell_mask,
            ctx_patch,
            cell_pos_in_ctx,
            item["label"],
            meta,
        )

    def get_labels(self) -> List[Any]:
        return [item["label"] for item in self.items]
