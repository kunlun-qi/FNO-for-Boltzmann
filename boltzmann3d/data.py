"""Strict MATLAB-v7.3 loading, deterministic splitting, and normalization.

MATLAB stores the tensors logically as ``[sample, channel, v1, v2, v3]``.
HDF5 exposes the dimensions in reverse order, so the raw arrays must be
transposed explicitly.  Avoiding shape heuristics prevents silent velocity-
axis permutations when the sample count or grid resolution changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class Normalization:
    """Statistics fitted on the training subset only."""

    f_mean: float
    f_std: float
    delta_mean: float
    delta_std: float
    velocity_scale: float

    def to_dict(self) -> dict[str, float]:
        return {
            "f_mean": self.f_mean,
            "f_std": self.f_std,
            "delta_mean": self.delta_mean,
            "delta_std": self.delta_std,
            "velocity_scale": self.velocity_scale,
        }


@dataclass(frozen=True)
class DataBundle:
    """Physical arrays, model tensors, and reproducible sample indices."""

    input_physical: np.ndarray
    target_physical: np.ndarray
    model_input: np.ndarray
    model_target: np.ndarray
    splits: dict[str, np.ndarray]
    normalization: Normalization
    velocity: tuple[np.ndarray, np.ndarray, np.ndarray]
    dv: float


def _read_matlab_tensor(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"{path} does not contain MATLAB variable {key!r}")
        raw = handle[key][:]

    if raw.ndim != 5:
        raise ValueError(f"Expected a five-dimensional tensor in {path}, got {raw.shape}")

    # HDF5: [v3,v2,v1,channel,sample] -> Python: [sample,channel,v1,v2,v3]
    return np.transpose(raw, (4, 3, 2, 1, 0)).astype(np.float32, copy=False)


def _make_stratified_splits(
    family_counts: list[int], split_counts: dict[str, list[int]], seed: int
) -> dict[str, np.ndarray]:
    required = ("train", "validation", "test")
    if any(name not in split_counts for name in required):
        raise ValueError(f"split_counts must define {required}")

    n_families = len(family_counts)
    if any(len(split_counts[name]) != n_families for name in required):
        raise ValueError("Every split must specify one count per initial-condition family")

    rng = np.random.default_rng(seed)
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in required}
    start = 0
    for family, family_count in enumerate(family_counts):
        requested = sum(split_counts[name][family] for name in required)
        if requested != family_count:
            raise ValueError(
                f"Family {family + 1}: split counts sum to {requested}, expected {family_count}"
            )

        indices = np.arange(start, start + family_count, dtype=np.int64)
        rng.shuffle(indices)
        offset = 0
        for name in required:
            count = split_counts[name][family]
            pieces[name].append(indices[offset : offset + count])
            offset += count
        start += family_count

    splits: dict[str, np.ndarray] = {}
    for name in required:
        merged = np.concatenate(pieces[name])
        rng.shuffle(merged)
        splits[name] = merged

    all_indices = np.concatenate([splits[name] for name in required])
    if len(np.unique(all_indices)) != sum(family_counts):
        raise RuntimeError("Train/validation/test splits are not disjoint and exhaustive")
    return splits


def _load_explicit_splits(
    path: Path,
    family_counts: list[int],
    split_counts: dict[str, list[int]],
) -> dict[str, np.ndarray]:
    """Load and validate an experiment's fixed stratified split."""

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    required = ("train", "validation", "test")
    splits = {
        name: np.asarray(raw[name], dtype=np.int64)
        for name in required
    }
    n_samples = sum(family_counts)
    merged = np.concatenate([splits[name] for name in required])
    if len(merged) != n_samples or set(merged.tolist()) != set(range(n_samples)):
        raise ValueError("Explicit splits must be disjoint and exhaustive")

    boundaries = np.cumsum([0, *family_counts])
    for name in required:
        actual = [
            int(np.sum((splits[name] >= boundaries[i]) & (splits[name] < boundaries[i + 1])))
            for i in range(len(family_counts))
        ]
        if actual != split_counts[name]:
            raise ValueError(
                f"Explicit {name} family allocation {actual} != {split_counts[name]}"
            )
    return splits


def load_data_bundle(project_dir: Path, config: dict[str, Any]) -> DataBundle:
    """Load configured endpoint pairs and prepare residual-learning tensors."""

    data_cfg = config["data"]
    x = _read_matlab_tensor(project_dir / data_cfg["input_file"], "input_data")
    y = _read_matlab_tensor(project_dir / data_cfg["target_file"], "target_data")

    expected_n = sum(data_cfg["family_counts"])
    n_grid = int(config["velocity"]["N"])
    if x.shape != (expected_n, 4, n_grid, n_grid, n_grid):
        raise ValueError(f"Unexpected input shape {x.shape}")
    if y.shape != (expected_n, 1, n_grid, n_grid, n_grid):
        raise ValueError(f"Unexpected target shape {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("The endpoint dataset contains non-finite values")

    if "split_indices_file" in data_cfg:
        splits = _load_explicit_splits(
            project_dir / data_cfg["split_indices_file"],
            data_cfg["family_counts"],
            data_cfg["split_counts"],
        )
    else:
        splits = _make_stratified_splits(
            data_cfg["family_counts"], data_cfg["split_counts"], int(config["seed"])
        )
    train = splits["train"]

    f0 = x[:, 3:4]
    delta = y - f0
    f_mean = float(np.mean(f0[train], dtype=np.float64))
    f_std = float(np.std(f0[train], dtype=np.float64))
    delta_mean = float(np.mean(delta[train], dtype=np.float64))
    delta_std = float(np.std(delta[train], dtype=np.float64))
    if f_std <= np.finfo(np.float32).eps or delta_std <= np.finfo(np.float32).eps:
        raise ValueError("Degenerate training normalization statistics")

    velocity_scale = float(config["velocity"]["L"])
    normalization = Normalization(
        f_mean=f_mean,
        f_std=f_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
        velocity_scale=velocity_scale,
    )

    model_input = np.empty_like(x)
    model_input[:, :3] = x[:, :3] / velocity_scale
    model_input[:, 3:4] = (f0 - f_mean) / f_std
    model_target = (delta - delta_mean) / delta_std

    velocity = tuple(x[0, axis].copy() for axis in range(3))
    return DataBundle(
        input_physical=x,
        target_physical=y,
        model_input=model_input,
        model_target=model_target,
        splits=splits,
        normalization=normalization,
        velocity=velocity,  # type: ignore[arg-type]
        dv=float(config["velocity"]["dv"]),
    )


def normalization_from_mapping(values: dict[str, Any]) -> Normalization:
    """Reconstruct saved normalization metadata from a checkpoint."""

    return Normalization(
        f_mean=float(values["f_mean"]),
        f_std=float(values["f_std"]),
        delta_mean=float(values["delta_mean"]),
        delta_std=float(values["delta_std"]),
        velocity_scale=float(values["velocity_scale"]),
    )
