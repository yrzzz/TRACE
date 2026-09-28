#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare registered Xenium coordinates, annotations, and polygons for TRACE.
See README.md for input formats and commands.
"""

import argparse
import json
import os
import tarfile
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import tifffile
import zarr


_XENIUM_LABEL_ALIASES = {
    "cluster": {"cluster"},
    "level_1": {"level_1", "level1", "level 1"},
    "level_2": {"level_2", "level2", "level 2"},
    "level_3": {"level_3", "level3", "level 3"},
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {v}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def pick_first_existing(columns: Iterable[str], candidates: Iterable[str], name: str) -> str:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Cannot find {name}. Tried: {list(candidates)} | available={sorted(cols)}")


def canonicalize_xenium_label_col(label_col: str) -> str:
    raw = str(label_col).strip().lower()
    for canonical, aliases in _XENIUM_LABEL_ALIASES.items():
        if raw in aliases:
            return canonical
    return raw


def transform_xy_to_he(
    x_values: np.ndarray,
    y_values: np.ndarray,
    minv: np.ndarray,
    xenium_mpp: float,
    input_is_micron: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)

    if input_is_micron:
        x = x / float(xenium_mpp)
        y = y / float(xenium_mpp)

    xy1 = np.stack([x, y, np.ones_like(x)], axis=1)
    uv1 = xy1 @ minv.T
    return uv1[:, 0].astype(np.float32), uv1[:, 1].astype(np.float32)


def load_clusters_from_tar(analysis_tar: str, clusters_inner: str) -> pd.DataFrame:
    with tarfile.open(analysis_tar, "r:gz") as tf:
        try:
            member = tf.getmember(clusters_inner)
        except KeyError:
            member = None
            normalized_target = clusters_inner.lstrip("./")
            for m in tf.getmembers():
                name = m.name.lstrip("./")
                if name == normalized_target or name.endswith("/" + normalized_target):
                    member = m
                    break
            if member is None:
                raise FileNotFoundError(f"Missing in tar: {clusters_inner}")
        fobj = tf.extractfile(member)
        if fobj is None:
            raise RuntimeError(f"Failed to extract {clusters_inner} from tar")
        clusters = pd.read_csv(fobj)

    clusters = normalize_columns(clusters)
    join_col = pick_first_existing(clusters.columns, ["barcode", "cell_id"], "cluster join key")
    cluster_col = pick_first_existing(clusters.columns, ["cluster"], "cluster label")
    out = clusters[[join_col, cluster_col]].copy()
    out = out.rename(columns={join_col: "join_id", cluster_col: "cluster"})
    out["join_id"] = out["join_id"].astype(str)
    out["cluster"] = pd.to_numeric(out["cluster"], errors="coerce").astype("Int64")
    out = out.drop_duplicates(subset=["join_id"], keep="first").reset_index(drop=True)
    return out


def _normalize_text_series(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA, "na": pd.NA})
    return out


def load_lung_annotations(annotation_csv: str) -> pd.DataFrame:
    ann = pd.read_csv(annotation_csv)
    ann = normalize_columns(ann)
    if ann.empty:
        raise ValueError(f"Annotation file is empty: {annotation_csv}")

    id_col = None
    id_candidates = ("cell_id", "barcode", "join_id", "unnamed: 0", "index")
    for c in id_candidates:
        if c in ann.columns:
            id_col = c
            break
    if id_col is None:
        id_col = str(ann.columns[0])

    out = pd.DataFrame({"join_id": ann[id_col].astype(str).str.strip()})

    found_levels = []
    for canonical in ("level_1", "level_2", "level_3"):
        src_col = None
        for c in _XENIUM_LABEL_ALIASES[canonical]:
            if c in ann.columns:
                src_col = c
                break
        if src_col is None:
            continue
        out[canonical] = _normalize_text_series(ann[src_col])
        found_levels.append(canonical)

    if not found_levels:
        raise ValueError(
            f"Annotation file must contain one of level columns "
            f"{sorted(_XENIUM_LABEL_ALIASES['level_1'] | _XENIUM_LABEL_ALIASES['level_2'] | _XENIUM_LABEL_ALIASES['level_3'])}"
        )

    out = out[out["join_id"].notna() & (out["join_id"] != "")]
    out = out.drop_duplicates(subset=["join_id"], keep="first").reset_index(drop=True)
    return out


def load_cells(cells_csv: str) -> pd.DataFrame:
    cells = pd.read_csv(cells_csv, compression="infer")
    cells = normalize_columns(cells)

    join_col = "barcode" if "barcode" in cells.columns else ("cell_id" if "cell_id" in cells.columns else None)
    if join_col is None:
        raise ValueError("cells.csv.gz must contain barcode or cell_id")

    x_col = pick_first_existing(cells.columns, ["x_centroid", "x", "centroid_x"], "centroid x")
    y_col = pick_first_existing(cells.columns, ["y_centroid", "y", "centroid_y"], "centroid y")
    cell_id_col = "cell_id" if "cell_id" in cells.columns else join_col

    # Build explicitly so this also works when join_col == cell_id_col.
    out = pd.DataFrame(
        {
            "join_id": cells[join_col].astype(str),
            "cell_id": cells[cell_id_col].astype(str),
            "centroid_x_raw": pd.to_numeric(cells[x_col], errors="coerce"),
            "centroid_y_raw": pd.to_numeric(cells[y_col], errors="coerce"),
        }
    )
    out = out[out["centroid_x_raw"].notna() & out["centroid_y_raw"].notna()].copy()
    return out


def _append_points(
    acc: Dict[str, Dict[str, list]],
    cell_ids: np.ndarray,
    x_he: np.ndarray,
    y_he: np.ndarray,
) -> None:
    for cid, x_val, y_val in zip(cell_ids, x_he, y_he):
        key = str(cid)
        if key not in acc:
            acc[key] = {"x": [], "y": []}
        acc[key]["x"].append(float(x_val))
        acc[key]["y"].append(float(y_val))


def load_boundary_points(
    bnd_csv: str,
    keep_cell_ids: set,
    minv: np.ndarray,
    xenium_mpp: float,
    input_is_micron: bool,
    chunksize: int,
) -> Dict[str, Dict[str, list]]:
    acc: Dict[str, Dict[str, list]] = {}
    n_rows_kept = 0

    reader = pd.read_csv(bnd_csv, compression="infer", chunksize=chunksize)
    for chunk in reader:
        chunk = normalize_columns(chunk)
        need = {"cell_id", "vertex_x", "vertex_y"}
        if not need.issubset(set(chunk.columns)):
            raise ValueError(f"Boundary chunk missing columns {need}, got {list(chunk.columns)}")

        chunk["cell_id"] = chunk["cell_id"].astype(str)
        chunk = chunk[chunk["cell_id"].isin(keep_cell_ids)]
        if chunk.empty:
            continue

        x_he, y_he = transform_xy_to_he(
            chunk["vertex_x"].to_numpy(),
            chunk["vertex_y"].to_numpy(),
            minv=minv,
            xenium_mpp=xenium_mpp,
            input_is_micron=input_is_micron,
        )
        _append_points(acc, chunk["cell_id"].to_numpy(), x_he, y_he)
        n_rows_kept += int(len(chunk))

    print(f"[INFO] kept boundary rows after filtering: {n_rows_kept}")
    return acc


def maybe_save_sanity_plot(
    he_tif: str,
    cells_prepared: pd.DataFrame,
    out_dir: str,
    max_points: int = 2000,
) -> str:
    import matplotlib.pyplot as plt

    if cells_prepared.empty:
        return ""

    def _resolve_zarr_array(node):
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

    with tifffile.TiffFile(he_tif) as tf:
        series = tf.series[0]
        levels = series.levels if len(series.levels) > 0 else [series]
        level0 = levels[0]
        thumb_level = levels[-1]
        level0_arr = _resolve_zarr_array(zarr.open(level0.aszarr(), mode="r"))
        thumb_arr = _resolve_zarr_array(zarr.open(thumb_level.aszarr(), mode="r"))
        if level0_arr is None or thumb_arr is None:
            raise ValueError("Failed to resolve zarr arrays for sanity plot.")

        if thumb_arr.ndim == 2:
            thumb = np.asarray(thumb_arr, dtype=np.uint8)
            thumb = np.stack([thumb] * 3, axis=-1)
        elif thumb_arr.ndim == 3 and thumb_arr.shape[-1] in (3, 4):
            thumb = np.asarray(thumb_arr[..., :3], dtype=np.uint8)
        elif thumb_arr.ndim == 3 and thumb_arr.shape[0] in (3, 4):
            thumb = np.asarray(np.moveaxis(thumb_arr[:3, ...], 0, -1), dtype=np.uint8)
        else:
            raise ValueError(f"Unexpected thumbnail shape: {thumb_arr.shape}")

        h0, w0 = int(level0_arr.shape[0]), int(level0_arr.shape[1])
        ht, wt = int(thumb.shape[0]), int(thumb.shape[1])
        sx = float(wt) / float(w0)
        sy = float(ht) / float(h0)

    n = min(max_points, len(cells_prepared))
    sample = cells_prepared.sample(n=n, random_state=0)
    x = sample["centroid_u"].to_numpy(dtype=np.float32) * sx
    y = sample["centroid_v"].to_numpy(dtype=np.float32) * sy

    out_path = os.path.join(out_dir, "sanity_overlay.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(thumb)
    plt.scatter(x, y, s=2, c="red", alpha=0.6)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def prepare_xenium_data(
    align_csv: str,
    he_tif: str,
    cells_csv: str,
    bnd_csv: str,
    analysis_tar: str,
    clusters_inner: str,
    annotation_csv: str,
    target_label_col: str,
    xenium_mpp: float,
    input_is_micron: bool,
    out_dir: str,
    chunksize: int,
    sanity_plot: bool = False,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)

    m = np.loadtxt(align_csv, delimiter=",")
    minv = np.linalg.inv(m)

    target_label_col = canonicalize_xenium_label_col(target_label_col)

    need_clusters = target_label_col == "cluster"
    if os.path.exists(analysis_tar):
        clusters = load_clusters_from_tar(analysis_tar, clusters_inner)
    elif need_clusters:
        raise FileNotFoundError(
            f"analysis_tar not found ({analysis_tar}) but target_label_col=cluster requires cluster labels."
        )
    else:
        print(f"[WARN] analysis_tar not found ({analysis_tar}); proceeding without cluster labels.")
        clusters = pd.DataFrame({"join_id": pd.Series(dtype=str), "cluster": pd.Series(dtype="Int64")})
    cells = load_cells(cells_csv)

    merged = cells.merge(clusters, on="join_id", how="left")
    if "cluster" in merged.columns:
        merged["cluster"] = pd.to_numeric(merged["cluster"], errors="coerce").astype("Int64")
        cluster_kept = int(merged["cluster"].notna().sum())
    else:
        cluster_kept = 0

    ann_path = str(annotation_csv).strip()
    ann_used = False
    if ann_path and os.path.exists(ann_path):
        ann = load_lung_annotations(ann_path)
        merged = merged.merge(ann, on="join_id", how="left")
        ann_used = True
        print(f"[INFO] merged annotation rows: {len(ann)} from {ann_path}")
    elif target_label_col in ("level_1", "level_2", "level_3"):
        raise FileNotFoundError(
            f"target_label_col={target_label_col} requested but annotation_csv not found: {ann_path}"
        )

    if target_label_col not in merged.columns:
        raise ValueError(
            f"target_label_col={target_label_col} not found after merge. "
            f"Available columns: {sorted(merged.columns.tolist())}"
        )

    before_drop = len(merged)
    merged = merged[merged[target_label_col].notna()].copy()
    print(f"[INFO] cells with non-NaN {target_label_col}: {len(merged)} / {before_drop}")
    label_kept_cells = int(len(merged))

    centroid_u, centroid_v = transform_xy_to_he(
        merged["centroid_x_raw"].to_numpy(),
        merged["centroid_y_raw"].to_numpy(),
        minv=minv,
        xenium_mpp=xenium_mpp,
        input_is_micron=input_is_micron,
    )
    merged["centroid_u"] = centroid_u
    merged["centroid_v"] = centroid_v

    keep_cell_ids = set(merged["cell_id"].astype(str).tolist())
    boundary_dict = load_boundary_points(
        bnd_csv=bnd_csv,
        keep_cell_ids=keep_cell_ids,
        minv=minv,
        xenium_mpp=xenium_mpp,
        input_is_micron=input_is_micron,
        chunksize=chunksize,
    )

    polygons = []
    poly_lookup: Dict[str, int] = {}
    for cid, points in boundary_dict.items():
        x = np.asarray(points["x"], dtype=np.float32)
        y = np.asarray(points["y"], dtype=np.float32)
        if x.size < 3 or y.size < 3:
            continue
        if x[0] != x[-1] or y[0] != y[-1]:
            x = np.concatenate([x, x[:1]], axis=0)
            y = np.concatenate([y, y[:1]], axis=0)
        poly = np.stack([y, x], axis=0).astype(np.float32)
        poly_lookup[cid] = len(polygons)
        polygons.append(poly)

    merged["poly_idx"] = merged["cell_id"].astype(str).map(poly_lookup)
    merged = merged[merged["poly_idx"].notna()].copy()
    merged["poly_idx"] = merged["poly_idx"].astype(int)
    label_cols = [c for c in ("cluster", "level_1", "level_2", "level_3") if c in merged.columns]
    merged = merged[["cell_id", "centroid_u", "centroid_v"] + label_cols + ["poly_idx"]]
    if "cluster" in merged.columns:
        merged["cluster"] = pd.to_numeric(merged["cluster"], errors="coerce").astype("Int64")
    for c in ("level_1", "level_2", "level_3"):
        if c in merged.columns:
            merged[c] = _normalize_text_series(merged[c])

    # Force a 1D object array; np.asarray(..., dtype=object) can still try to
    # broadcast when each polygon starts with the same leading dim (2, P_i).
    polygons_arr = np.empty((len(polygons),), dtype=object)
    for i, poly in enumerate(polygons):
        polygons_arr[i] = poly
    polygons_path = os.path.join(out_dir, "polygons.npy")
    cells_path = os.path.join(out_dir, "cells_prepared.csv")
    np.save(polygons_path, polygons_arr, allow_pickle=True)
    merged.to_csv(cells_path, index=False)

    if len(merged) == 0:
        raise RuntimeError("No cells retained after cluster/poly filtering.")
    if not np.isfinite(merged[["centroid_u", "centroid_v"]].to_numpy(dtype=np.float32)).all():
        raise RuntimeError("Found non-finite centroid values in prepared cells.")
    if len(polygons_arr) == 0 or min(p.shape[1] for p in polygons_arr) < 3:
        raise RuntimeError("Invalid polygon list: requires at least 3 vertices per polygon.")
    if merged["poly_idx"].min() < 0 or merged["poly_idx"].max() >= len(polygons_arr):
        raise RuntimeError("poly_idx out of bounds against saved polygons.npy")
    poly_lengths = np.asarray([int(p.shape[1]) for p in polygons_arr], dtype=np.int32)
    if (poly_lengths < 3).any():
        raise RuntimeError("Found polygon with <3 points.")

    sanity_path = ""
    if sanity_plot:
        sanity_path = maybe_save_sanity_plot(he_tif=he_tif, cells_prepared=merged, out_dir=out_dir)

    meta = {
        "align_csv": align_csv,
        "he_tif": he_tif,
        "cells_csv": cells_csv,
        "bnd_csv": bnd_csv,
        "analysis_tar": analysis_tar,
        "clusters_inner": clusters_inner,
        "xenium_mpp": float(xenium_mpp),
        "input_is_micron": bool(input_is_micron),
        "num_cells_cluster_kept": int(cluster_kept),
        "num_cells_label_kept": int(label_kept_cells),
        "target_label_col": target_label_col,
        "annotation_csv": ann_path if ann_used else "",
        "num_cells_final": int(len(merged)),
        "num_polygons": int(len(polygons_arr)),
        "chunksize": int(chunksize),
        "sanity_overlay": sanity_path,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVE] {cells_path}")
    print(f"[SAVE] {polygons_path}")
    print(f"[INFO] cells_prepared shape: {merged.shape}")
    print(
        "[INFO] polygons count/min/median/max points: "
        f"{len(polygons_arr)}/{int(poly_lengths.min())}/{float(np.median(poly_lengths)):.1f}/{int(poly_lengths.max())}"
    )
    print(
        "[CHECK] finite centroids="
        f"{bool(np.isfinite(merged[['centroid_u', 'centroid_v']].to_numpy(dtype=np.float32)).all())}, "
        f"poly_idx_bounds=[{int(merged['poly_idx'].min())}, {int(merged['poly_idx'].max())}]"
    )
    if sanity_path:
        print(f"[SAVE] {sanity_path}")
    return meta


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align_csv", default="data/Xenium_V1_Human_Lung_Cancer_FFPE_he_imagealignment.csv")
    ap.add_argument("--he_tif", default="data/Xenium_V1_Human_Lung_Cancer_FFPE_he_image.ome.tif")
    ap.add_argument("--cells_csv", default="data/cells.csv.gz")
    ap.add_argument("--bnd_csv", default="data/nucleus_boundaries.csv.gz")
    ap.add_argument("--analysis_tar", default="data/analysis.tar.gz")
    ap.add_argument(
        "--clusters_inner",
        default="analysis/clustering/gene_expression_kmeans_3_clusters/clusters.csv",
    )
    ap.add_argument("--annotation_csv", default="data/lung_cancer_annotation.csv")
    ap.add_argument("--target_label_col", default="cluster")
    ap.add_argument("--xenium_mpp", type=float, default=0.2125)
    ap.add_argument("--input_is_micron", type=str2bool, default=True)
    ap.add_argument("--out_dir", default="data/xenium_prepared")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--sanity_plot", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    prepare_xenium_data(
        align_csv=args.align_csv,
        he_tif=args.he_tif,
        cells_csv=args.cells_csv,
        bnd_csv=args.bnd_csv,
        analysis_tar=args.analysis_tar,
        clusters_inner=args.clusters_inner,
        annotation_csv=args.annotation_csv,
        target_label_col=args.target_label_col,
        xenium_mpp=args.xenium_mpp,
        input_is_micron=args.input_is_micron,
        out_dir=args.out_dir,
        chunksize=args.chunksize,
        sanity_plot=args.sanity_plot,
    )


if __name__ == "__main__":
    main()
