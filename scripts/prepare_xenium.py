"""Prepare Xenium from the same config used by train.py."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_runtime import load_config
from xenium_prepare import prepare_xenium_data, str2bool


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg["dataset"] != "xenium":
        raise ValueError("This preparation command requires dataset=xenium")
    out = Path(cfg["prepared_dir"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Prepared directory is not empty: {out}. Choose a new prepared_dir.")
    prepare_xenium_data(
        align_csv=cfg["align_csv"], he_tif=cfg["he_tif"], cells_csv=cfg["cells_csv"],
        bnd_csv=cfg["bnd_csv"], analysis_tar="", clusters_inner="",
        annotation_csv=cfg["annotation_csv"], target_label_col=cfg["xenium_label_level"],
        xenium_mpp=cfg.get("xenium_mpp", 0.2125),
        input_is_micron=str2bool(cfg.get("input_is_micron", True)), out_dir=str(out),
        chunksize=cfg.get("chunksize", 1000000), sanity_plot=False,
    )


if __name__ == "__main__":
    main()
