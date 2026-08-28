#!/usr/bin/env python3
"""Validate the hybrid dataset and record its measured generation time."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from boltzmann3d.data import _read_matlab_tensor


PROJECT = Path(__file__).resolve().parent


def main() -> None:
    config_path = PROJECT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_dir = PROJECT / "data"
    old_x = _read_matlab_tensor(data_dir / "input_data_3V_baseline50.mat", "input_data")
    old_y = _read_matlab_tensor(data_dir / "target_data_3V_baseline50.mat", "target_data")
    new_x = _read_matlab_tensor(data_dir / "input_data_3V.mat", "input_data")
    new_y = _read_matlab_tensor(data_dir / "target_data_3V.mat", "target_data")

    if new_x.shape != (50, 4, 32, 32, 32) or new_y.shape != (50, 1, 32, 32, 32):
        raise ValueError(f"Unexpected hybrid shapes: {new_x.shape}, {new_y.shape}")
    if not np.isfinite(new_x).all() or not np.isfinite(new_y).all():
        raise ValueError("Hybrid endpoint tensors contain non-finite values")

    unchanged = np.r_[0:16, 33:50]
    np.testing.assert_array_equal(new_x[unchanged], old_x[unchanged])
    np.testing.assert_array_equal(new_y[unchanged], old_y[unchanged])
    if np.array_equal(new_x[16:33, 3], old_x[16:33, 3]):
        raise ValueError("The BKW inputs were not replaced")

    metadata_path = data_dir / "hybrid_bkw_metadata.mat"
    with h5py.File(metadata_path, "r") as handle:
        subtype = handle["bkw_subtype"][:].reshape(-1).astype(int)
        start_time = handle["bkw_paper_start_time"][:].reshape(-1)
        epsilon = handle["bkw_perturbation_epsilon"][:].reshape(-1)
        generation_seconds = float(handle["generation_seconds"][0, 0])
    if np.count_nonzero(subtype == 1) != 9 or np.count_nonzero(subtype == 2) != 8:
        raise ValueError("Hybrid subtype counts are not 9 exact plus 8 perturbed")
    if np.any(start_time[16:25] <= config["hybrid_bkw"]["paper_time_positivity_threshold"]):
        raise ValueError("An exact BKW start time violates positivity")
    if not np.all((epsilon[25:33] > 0) & (epsilon[25:33] <= 0.1)):
        raise ValueError("A perturbed BKW amplitude is outside (0,0.1]")
    if np.any(new_x[16:33, 3] < 0):
        raise ValueError("A hybrid BKW training input is negative")

    benchmark_x = _read_matlab_tensor(
        data_dir / "bkw_benchmark_3d.mat", "input_data"
    )[0, 3]
    if any(np.array_equal(sample, benchmark_x) for sample in new_x[:, 3]):
        raise ValueError("The reserved exact f_BKW(5.5) benchmark entered training")

    splits = json.loads((data_dir / "fixed_split_indices.json").read_text(encoding="utf-8"))
    train_bkw = [index for index in splits["train"] if 16 <= index < 33]
    train_subtypes = {int(subtype[index]) for index in train_bkw}
    if train_subtypes != {1, 2} or len(train_bkw) != 13:
        raise ValueError("The fixed training split must contain both BKW subtypes and 13 BKW members")

    baseline_seconds = float(config["data"]["baseline_generation_seconds"])
    config["data"]["hybrid_replacement_generation_seconds"] = generation_seconds
    config["data"]["generation_seconds"] = baseline_seconds + generation_seconds
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Validated exact preservation of {len(unchanged)} non-BKW pairs")
    print(f"Validated 9 exact plus 8 perturbed BKW pairs; {len(train_bkw)} enter training")
    print("Validated exclusion of the exact f_BKW(5.5) benchmark input")
    print(f"Recorded hybrid replacement time: {generation_seconds:.1f} s")
    print(
        "Recorded total training-data cost: "
        f"{baseline_seconds + generation_seconds:.1f} s "
        f"({baseline_seconds:.1f} s inherited baseline + replacements)"
    )


if __name__ == "__main__":
    main()
