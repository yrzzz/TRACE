#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tifffile
import zarr
from skimage.draw import polygon as sk_polygon
from torch.utils.data import Dataset

from datasets.skin_cells import CellMeta


def _ensure_closed_polygon(poly_yx: np.ndarray) -> np.ndarray:
    if poly_yx.shape[1] < 3:
        return poly_yx
    if poly_yx[0, 0] == poly_yx[0, -1] and poly_yx[1, 0] == poly_yx[1, -1]:
        return poly_yx
    return np.concatenate([poly_yx, poly_yx[:, :1]], axis=1)


def _resolve_zarr_array(node):
    """Resolve first array from a zarr node that may be an Array or Group."""
    try:
        _ = tuple(node.shape)
        return node
    except Exception:
        pass

    if hasattr(node, "array_keys"):
        for k in sorted(list(node.array_keys())):
            arr = _resolve_zarr_array(node[k])
            if arr is not None:
                return arr

    if hasattr(node, "group_keys"):
        for k in sorted(list(node.group_keys())):
            arr = _resolve_zarr_array(node[k])
            if arr is not None:
                return arr

    if hasattr(node, "keys"):
        for k in sorted(list(node.keys())):
            arr = _resolve_zarr_array(node[k])
            if arr is not None:
                return arr

    return None


class XeniumDataset(Dataset):
    def __init__(
        self,
        prepared_dir: str,
        he_tif: str = "",
        cell_size: int = 256,
        ctx_size: int = 1024,
        skip_ctx: bool = False,
        label_col: str = "cluster",
        sample_name: str = "xenium",
    ) -> None:
        self.prepared_dir = prepared_dir
        cells_path = os.path.join(prepared_dir, "cells_prepared.csv")
        polygons_path = os.path.join(prepared_dir, "polygons.npy")
        meta_path = os.path.join(prepared_dir, "meta.json")
        if not os.path.exists(cells_path):
            raise FileNotFoundError(f"Missing prepared file: {cells_path}")
        if not os.path.exists(polygons_path):
            raise FileNotFoundError(f"Missing prepared file: {polygons_path}")

        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        self.he_tif = he_tif if he_tif else str(meta.get("he_tif", ""))
        if not self.he_tif:
            raise ValueError("HE OME-TIF path is required (pass he_tif or include it in prepared meta.json)")
        if not os.path.isabs(self.he_tif):
            # Keep relative paths resolved from repo root when running scripts from skin_cell_experts/.
            self.he_tif = os.path.normpath(self.he_tif)
        if not os.path.exists(self.he_tif):
            raise FileNotFoundError(f"HE OME-TIF not found: {self.he_tif}")

        self.df = pd.read_csv(cells_path)
        required_cols = {"cell_id", "centroid_u", "centroid_v", "poly_idx", label_col}
        missing = required_cols.difference(set(self.df.columns))
        if missing:
            raise ValueError(f"cells_prepared.csv missing columns: {sorted(missing)}")

        self.df["cell_id"] = self.df["cell_id"].astype(str)
        self.df = self.df[np.isfinite(self.df["centroid_u"]) & np.isfinite(self.df["centroid_v"])].copy()
        self.df = self.df[self.df["poly_idx"].notna()].copy()
        self.df["poly_idx"] = self.df["poly_idx"].astype(int)
        self.df = self.df.reset_index(drop=True)

        self.polygons = np.load(polygons_path, allow_pickle=True)
        self.cell_size = int(cell_size)
        self.ctx_size = int(ctx_size)
        self.skip_ctx = bool(skip_ctx)
        self.label_col = label_col
        self.sample_name = sample_name

        self._tif = None
        self._store = None
        self._arr = None
        self._layout = None
        self._h = None
        self._w = None

        self.items: List[Dict[str, Any]] = []
        for _, row in self.df.iterrows():
            self.items.append({"sample": self.sample_name, "cell_id": row["cell_id"]})

        print(f"[INFO] Xenium prepared cells: {len(self.df)} | polygons: {len(self.polygons)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tif"] = None
        state["_store"] = None
        state["_arr"] = None
        return state

    def _open_reader(self) -> None:
        if self._arr is not None:
            return
        self._tif = tifffile.TiffFile(self.he_tif)
        series = self._tif.series[0]
        level0 = series.levels[0] if len(series.levels) > 0 else series
        self._store = level0.aszarr()
        node = zarr.open(self._store, mode="r")
        self._arr = _resolve_zarr_array(node)
        if self._arr is None:
            raise ValueError(f"Failed to resolve zarr array from level0 store for {self.he_tif}")

        shape = tuple(self._arr.shape)
        if len(shape) == 2:
            self._layout = "yx"
            self._h, self._w = shape
        elif len(shape) == 3 and shape[-1] in (1, 3, 4):
            self._layout = "yxc"
            self._h, self._w = shape[0], shape[1]
        elif len(shape) == 3 and shape[0] in (1, 3, 4):
            self._layout = "cyx"
            self._h, self._w = shape[1], shape[2]
        else:
            raise ValueError(f"Unsupported OME-TIF array shape: {shape}")

    def _read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        self._open_reader()
        if self._layout == "yx":
            patch = np.asarray(self._arr[y0:y1, x0:x1], dtype=np.float32)
            patch = np.stack([patch] * 3, axis=-1)
        elif self._layout == "yxc":
            patch = np.asarray(self._arr[y0:y1, x0:x1, :], dtype=np.float32)
            if patch.shape[2] == 1:
                patch = np.repeat(patch, 3, axis=2)
            elif patch.shape[2] > 3:
                patch = patch[:, :, :3]
        else:
            patch = np.asarray(self._arr[:, y0:y1, x0:x1], dtype=np.float32)
            if patch.shape[0] == 1:
                patch = np.repeat(patch, 3, axis=0)
            elif patch.shape[0] > 3:
                patch = patch[:3, ...]
            patch = np.moveaxis(patch, 0, -1)

        patch = np.clip(patch, 0, 255).astype(np.uint8)
        return patch

    def _crop_square_patch(self, cx: float, cy: float, side: int) -> Tuple[np.ndarray, int, int]:
        self._open_reader()
        side = max(int(side), 2)
        x0 = int(round(float(cx) - side / 2.0))
        y0 = int(round(float(cy) - side / 2.0))
        x1 = x0 + side
        y1 = y0 + side

        ix0 = max(0, x0)
        iy0 = max(0, y0)
        ix1 = min(self._w, x1)
        iy1 = min(self._h, y1)

        if ix1 <= ix0 or iy1 <= iy0:
            return np.full((side, side, 3), 255, dtype=np.uint8), x0, y0

        patch = self._read_region(iy0, iy1, ix0, ix1)
        pt = iy0 - y0
        pl = ix0 - x0
        pb = y1 - iy1
        pr = x1 - ix1
        if pt or pb or pl or pr:
            patch = np.pad(patch, ((pt, pb), (pl, pr), (0, 0)), mode="edge")
        return patch.astype(np.uint8), x0, y0

    def _rasterize_mask(self, poly_yx: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
        poly_yx = _ensure_closed_polygon(np.asarray(poly_yx, dtype=np.float32))
        y_rel = poly_yx[0, :] - float(y0)
        x_rel = poly_yx[1, :] - float(x0)
        rr, cc = sk_polygon(y_rel, x_rel, shape=(side, side))
        mask = np.zeros((side, side), dtype=np.uint8)
        mask[rr, cc] = 1
        return mask

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        cx = float(row["centroid_u"])
        cy = float(row["centroid_v"])
        poly_idx = int(row["poly_idx"])
        if poly_idx < 0 or poly_idx >= len(self.polygons):
            raise IndexError(f"poly_idx out of range: {poly_idx} / {len(self.polygons)}")

        cell_patch, cell_x0, cell_y0 = self._crop_square_patch(cx=cx, cy=cy, side=self.cell_size)
        poly_yx = np.asarray(self.polygons[poly_idx], dtype=np.float32)
        if poly_yx.shape[0] != 2:
            raise ValueError(f"Polygon must have shape (2,P). Got {poly_yx.shape}")
        cell_mask = self._rasterize_mask(poly_yx=poly_yx, x0=cell_x0, y0=cell_y0, side=self.cell_size)

        if self.skip_ctx:
            ctx_patch = np.zeros((1, 1, 3), dtype=np.uint8)
            cell_pos_in_ctx = (0.0, 0.0)
        else:
            ctx_patch, ctx_x0, ctx_y0 = self._crop_square_patch(cx=cx, cy=cy, side=self.ctx_size)
            cell_pos_in_ctx = (float(cx - ctx_x0), float(cy - ctx_y0))

        meta = CellMeta(sample=self.sample_name, cell_id=row["cell_id"], poly_idx=poly_idx)
        label = row[self.label_col]
        return cell_patch, cell_mask, ctx_patch, cell_pos_in_ctx, label, meta

    def get_labels(self) -> List[Any]:
        return self.df[self.label_col].tolist()
