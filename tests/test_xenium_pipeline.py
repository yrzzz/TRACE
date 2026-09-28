"""Exercise TIFF/CSV registration, annotation matching, and binary label mapping."""
import gzip
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile

import trace_runtime as runtime
from xenium_prepare import prepare_xenium_data


class XeniumPipelineTests(unittest.TestCase):
    def test_prepare_crop_label_and_binary_remap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tifffile.imwrite(root / "he.tif", np.full((64, 64, 3), 127, dtype=np.uint8))
            np.savetxt(root / "alignment.csv", np.eye(3), delimiter=",")
            ids = [f"cell{i}-1" for i in range(14)]
            pd.DataFrame({"cell_id": ids, "x_centroid": 30., "y_centroid": 30.}).to_csv(root / "cells.csv.gz", index=False)
            boundary = [{"cell_id": cid, "vertex_x": x, "vertex_y": y}
                        for cid in ids for x, y in [(25, 25), (35, 25), (35, 35), (25, 35)]]
            pd.DataFrame(boundary).to_csv(root / "boundaries.csv.gz", index=False)
            # Two unannotated cells should be dropped instead of assigned a spurious class.
            pd.DataFrame({"cell_id": ids[:12], "level2": ["A", "B"] * 6}).to_csv(root / "labels.csv.gz", index=False)
            prepare_xenium_data(str(root / "alignment.csv"), str(root / "he.tif"),
                str(root / "cells.csv.gz"), str(root / "boundaries.csv.gz"), "", "",
                str(root / "labels.csv.gz"), "level_2", 1., False, str(root / "prepared"), 4)
            cfg = {**runtime.DEFAULTS, "prepared_dir": str(root / "prepared"), "he_tif": str(root / "he.tif"),
                   "cell_size": 32, "ctx_size": 48, "cell_type": "A"}
            dataset = runtime.make_dataset(cfg)
            self.assertEqual(len(dataset), 12)
            sample = dataset[0]
            self.assertEqual(sample[0].shape, (32, 32, 3))
            self.assertEqual(sample[2].shape, (48, 48, 3))
            self.assertGreater(sample[1].sum(), 0)
            self.assertEqual(sample[4], "target")
            self.assertEqual(set(dataset.get_labels()), {"target", "others"})
            self.assertEqual(set(runtime.dataset_manifest(dataset)["label"]), {"target", "others"})


if __name__ == "__main__":
    unittest.main()
