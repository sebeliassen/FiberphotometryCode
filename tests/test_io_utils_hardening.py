#!/usr/bin/env python3
"""
test_io_utils_hardening.py
=============================================================================
Unit tests for hardened core.io_utils routines:
- copy_dataset_bounded: scalar, small, and chunked datasets with attributes.
- inspect_checkpoint_iterations: non-numeric key resilience.
- compact_kpms_checkpoint: preservation of leading zeros and non-numeric keys.
=============================================================================
"""

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from core.io_utils import (
    compact_kpms_checkpoint,
    copy_dataset_bounded,
    inspect_checkpoint_iterations,
)


class TestIOUtilsHardening(unittest.TestCase):
    def test_copy_dataset_bounded_scalar_and_attributes(self):
        with tempfile.TemporaryDirectory() as td:
            src_p = Path(td) / "src.h5"
            dst_p = Path(td) / "dst.h5"
            with h5py.File(src_p, "w") as f_src:
                # 1. Scalar dataset
                ds_scalar = f_src.create_dataset("scalar_val", data=np.float64(3.14159))
                ds_scalar.attrs["unit"] = "meters"
                ds_scalar.attrs["version"] = 1

                # 2. Small array dataset
                ds_small = f_src.create_dataset("small_arr", data=np.arange(100, dtype=np.int32))
                ds_small.attrs["description"] = "indices"

                # 3. Chunked array dataset
                ds_chunked = f_src.create_dataset(
                    "chunked_arr",
                    data=np.ones((200, 50), dtype=np.float32),
                    chunks=(50, 50),
                )
                ds_chunked.attrs["dtype_info"] = "float32"

            with h5py.File(src_p, "r") as f_src, h5py.File(dst_p, "w") as f_dst:
                dst_fd = f_dst.id.get_vfd_handle()
                # Copy scalar
                copy_dataset_bounded(f_src["scalar_val"], f_dst, "scalar_val", dst_fd)
                self.assertIn("scalar_val", f_dst)
                self.assertEqual(f_dst["scalar_val"][()], np.float64(3.14159))
                self.assertEqual(f_dst["scalar_val"].attrs["unit"], "meters")
                self.assertEqual(f_dst["scalar_val"].attrs["version"], 1)

                # Copy small
                copy_dataset_bounded(f_src["small_arr"], f_dst, "small_arr", dst_fd)
                np.testing.assert_array_equal(f_dst["small_arr"][:], np.arange(100, dtype=np.int32))
                self.assertEqual(f_dst["small_arr"].attrs["description"], "indices")

                # Copy chunked
                copy_dataset_bounded(f_src["chunked_arr"], f_dst, "chunked_arr", dst_fd)
                np.testing.assert_array_equal(f_dst["chunked_arr"][:], np.ones((200, 50), dtype=np.float32))
                self.assertEqual(f_dst["chunked_arr"].attrs["dtype_info"], "float32")

    def test_inspect_checkpoint_iterations_ignores_non_numeric(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.h5"
            with h5py.File(p, "w") as f:
                snaps = f.create_group("model_snapshots")
                snaps.create_group("0")
                snaps.create_group("10")
                snaps.create_group("20")
                snaps.create_group("init_state")
                snaps.create_group("final_eval")

            iters = inspect_checkpoint_iterations(p)
            self.assertEqual(iters, [0, 10, 20])

    def test_compact_kpms_checkpoint_preserves_leading_zeros_and_nonnumeric(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.h5"
            with h5py.File(p, "w") as f:
                f.attrs["project"] = "kpms_test"
                meta = f.create_group("metadata")
                meta.create_dataset("fps", data=60.0)

                snaps = f.create_group("model_snapshots")
                # Groups with leading zeros
                g1 = snaps.create_group("001")
                g1.attrs["tag"] = "step_1"
                g1.create_dataset("weights", data=np.array([1.0, 2.0]))

                g2 = snaps.create_group("010")
                g2.attrs["tag"] = "step_10"
                g2.create_dataset("weights", data=np.array([3.0, 4.0]))

                # Non-numeric snapshot group
                gnon = snaps.create_group("baseline")
                gnon.attrs["tag"] = "base"
                gnon.create_dataset("prior", data=np.array([0.5]))

            # Run compaction keeping iteration 10
            compacted_path = compact_kpms_checkpoint(p, keep_iterations=[10])
            self.assertTrue(compacted_path.exists())

            with h5py.File(compacted_path, "r") as f:
                self.assertTrue(f.attrs.get("is_packed", False))
                self.assertEqual(f["metadata/fps"][()], 60.0)

                snaps = f["model_snapshots"]
                # Must retain exact string name "010"
                self.assertIn("010", snaps)
                self.assertNotIn("001", snaps)  # Pruned
                self.assertIn("baseline", snaps)  # Preserved non-numeric
                self.assertEqual(snaps["010"].attrs["tag"], "step_10")
                np.testing.assert_array_equal(snaps["010/weights"][:], np.array([3.0, 4.0]))
                self.assertEqual(snaps["baseline"].attrs["tag"], "base")
                np.testing.assert_array_equal(snaps["baseline/prior"][:], np.array([0.5]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
