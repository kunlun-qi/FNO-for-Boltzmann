"""Shared configuration, device, checkpoint, and prediction helpers."""

from __future__ import annotations

import json
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import Normalization, normalization_from_mapping
from .models import FNO3d, build_model


def project_directory() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    project_dir = project_directory()
    config_path = Path(path).resolve() if path else project_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return project_dir, json.load(handle)


def choose_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconstruct_distribution(
    normalized_delta: torch.Tensor, initial: torch.Tensor, norm: Normalization
) -> torch.Tensor:
    """Convert the learned normalized increment into the physical endpoint."""

    return initial + normalized_delta * norm.delta_std + norm.delta_mean


def model_parameter_count(model: torch.nn.Module) -> int:
    """Number of tensor elements (a complex element counts once)."""

    return sum(parameter.numel() for parameter in model.parameters())


def model_real_scalar_parameter_count(model: torch.nn.Module) -> int:
    """Number of stored real scalars (a complex element counts twice)."""

    return sum(
        parameter.numel() * (2 if torch.is_complex(parameter) else 1)
        for parameter in model.parameters()
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_manifest(project_dir: Path) -> dict[str, Any]:
    """Hash the core source, configuration, and immutable endpoint tensors.

    This is a snapshot of files at the time it is called.  It is not a claim
    that a checkpoint was trained from files later edited in place.
    """

    candidate_paths = [
        "config.json",
        "requirements.txt",
        "README.md",
        "run_benchmark.sh",
        "train.py",
        "evaluate.py",
        "benchmark_inference.py",
        "prepare_hybrid_data.py",
        "scripts/make_data_manifest.py",
        "boltzmann3d/data.py",
        "boltzmann3d/__init__.py",
        "boltzmann3d/models.py",
        "boltzmann3d/physics.py",
        "boltzmann3d/runtime.py",
        "boltzmann3d/training.py",
        "tests/test_pipeline.py",
        "tests/test_refined_training.py",
        "matlab/generate_baseline_training_3d.m",
        "matlab/generate_hybrid_training_3d.m",
        "matlab/generate_bkw_benchmarks_hybrid_3d.m",
        "matlab/generate_external_test_3d.m",
        "matlab/benchmark_spectral_3d.m",
        "matlab/fast_spectral/CBoltz3_fast_sph.m",
        "matlab/fast_spectral/precpt_fast_sph.m",
        "matlab/fast_spectral/int_F.m",
        "matlab/fast_spectral/getSphericalDesign.m",
        "matlab/fast_spectral/lgwt.m",
        "matlab/fast_spectral/ss007.00032.txt",
        "data/input_data_3V.mat",
        "data/target_data_3V.mat",
        "data/input_data_3V_baseline50.mat",
        "data/target_data_3V_baseline50.mat",
        "data/fixed_split_indices.json",
        "data/bkw_benchmark_3d.mat",
        "data/bkw_time_grid_3d.mat",
        "data/hybrid_bkw_metadata.mat",
        "data/external_test_data_3V.mat",
    ]
    relative_paths = [
        relative for relative in candidate_paths if (project_dir / relative).exists()
    ]
    return {
        "manifest_kind": "source_and_data_snapshot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {
            relative: file_sha256(project_dir / relative) for relative in relative_paths
        },
    }


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[FNO3d, Normalization, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    norm = normalization_from_mapping(checkpoint["normalization"])
    return model, norm, checkpoint
