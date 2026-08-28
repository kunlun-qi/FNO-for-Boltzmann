#!/usr/bin/env python3
"""Evaluate shared samples and the exact paper-time BKW 5.5-to-6.5 map."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from boltzmann3d.data import load_data_bundle
from boltzmann3d.physics import distribution_errors, numpy_moments, relative_l2
from boltzmann3d.runtime import (
    choose_device,
    file_sha256,
    load_checkpoint,
    load_config,
    reconstruct_distribution,
)


def family_of(index: int, counts: list[int]) -> int:
    boundary = 0
    for family, count in enumerate(counts, start=1):
        boundary += count
        if index < boundary:
            return family
    raise IndexError(index)


def negative_mass(field: np.ndarray, dv: float) -> float:
    return float(np.maximum(-np.asarray(field, dtype=np.float64), 0).sum() * dv**3)


def predict_numpy(
    model: torch.nn.Module,
    model_input: np.ndarray,
    initial: np.ndarray,
    normalization,
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(model_input)).to(device)
    f0 = torch.from_numpy(np.ascontiguousarray(initial)).to(device)
    with torch.inference_mode():
        prediction = reconstruct_distribution(model(x), f0, normalization)
    return prediction.cpu().numpy()


def sample_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    initial: np.ndarray,
    velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
    dv: float,
) -> dict[str, float]:
    errors = distribution_errors(prediction, reference, dv)
    mass_p, bulk_p, energy_p = numpy_moments(prediction, velocity, dv)
    mass_0, bulk_0, energy_0 = numpy_moments(initial, velocity, dv)
    errors.update(
        {
            "relative_L2": relative_l2(prediction, reference),
            "mass_drift": abs(mass_p - mass_0),
            "momentum_drift": float(
                np.linalg.norm(mass_p * bulk_p - mass_0 * bulk_0)
            ),
            "bulk_velocity_drift": float(np.linalg.norm(bulk_p - bulk_0)),
            "energy_drift": abs(energy_p - energy_0),
            "minimum": float(np.min(prediction)),
            "negative_mass": negative_mass(prediction, dv),
        }
    )
    return errors


def summarize_rows(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> dict[str, Any]:
    metric_fields = (
        "L1",
        "L2",
        "Linf",
        "relative_L2",
        "mass_drift",
        "momentum_drift",
        "bulk_velocity_drift",
        "energy_drift",
        "minimum",
        "negative_mass",
    )
    summary: dict[str, Any] = {}
    keys = sorted({tuple(row[field] for field in group_fields) for row in rows})
    for key in keys:
        selected = [
            row for row in rows if tuple(row[field] for field in group_fields) == key
        ]
        label = "/".join(str(item) for item in key)
        metrics = {}
        for field in metric_fields:
            values = [row[field] for row in selected]
            metrics[field] = {"mean": float(np.mean(values))}
            if field == "minimum":
                metrics[field]["min"] = float(np.min(values))
            else:
                metrics[field]["max"] = float(np.max(values))
        summary[label] = {"count": len(selected), **metrics}
    return summary


def validate_checkpoint(
    checkpoint: dict[str, Any],
    kind: str,
    config: dict[str, Any],
    bundle,
) -> None:
    """Reject stale checkpoints whose scientific setup differs from this run."""

    if checkpoint.get("model_kind") != kind:
        raise RuntimeError(f"Expected a {kind} checkpoint")
    if checkpoint.get("model_config") != config["model"]:
        raise RuntimeError(f"{kind} checkpoint model configuration is stale")
    saved_splits = checkpoint.get("splits", {})
    current_splits = {name: values.tolist() for name, values in bundle.splits.items()}
    if saved_splits != current_splits:
        raise RuntimeError(f"{kind} checkpoint data split is stale")
    for name, value in bundle.normalization.to_dict().items():
        if not np.isclose(float(checkpoint["normalization"][name]), value, rtol=1e-7):
            raise RuntimeError(f"{kind} checkpoint normalization {name} is stale")

    # Lambda is irrelevant to the plain FNO objective.  Every other training
    # setting must match; C-FNO must also carry the frozen current lambda.
    ignored = (
        {"lambda_conservation", "cfno_conservation_alpha"}
        if kind == "fno"
        else set()
    )
    for name, value in config["training"].items():
        if name not in ignored and checkpoint["training_config"].get(name) != value:
            raise RuntimeError(f"{kind} checkpoint training setting {name} is stale")


def checkpoint_metadata(
    path: Path, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    return {
        "path": str(Path("checkpoints") / path.name),
        "sha256": file_sha256(path),
        "epoch": int(checkpoint["epoch"]),
        "model_config": checkpoint["model_config"],
        "training_config": checkpoint["training_config"],
        "validation_metrics": checkpoint["validation_metrics"],
        "embedded_training_provenance_available": any(
            key in checkpoint for key in ("provenance", "training_run_start_snapshot")
        ),
    }


def evaluate_heldout(
    project_dir: Path, config: dict[str, Any], device: torch.device
) -> None:
    bundle = load_data_bundle(project_dir, config)
    external_path = project_dir / "data" / "external_test_data_3V.mat"
    if not external_path.exists():
        raise FileNotFoundError(
            "Copy or generate the shared external comparison set."
        )
    with h5py.File(external_path, "r") as handle:
        physical_input = np.transpose(
            handle["input_data"][:], (4, 3, 2, 1, 0)
        ).astype(np.float32, copy=False)
        target_physical = np.transpose(
            handle["target_data"][:], (4, 3, 2, 1, 0)
        ).astype(np.float32, copy=False)
        family_ids = handle["family_id"][:].reshape(-1).astype(int)
        dv = float(handle["dv"][0, 0])
        external_seed = int(handle["random_seed"][0, 0])
        dt = float(handle["dt"][0, 0])
        time_horizon = float(handle["time_horizon"][0, 0])
        family_parameters = np.asarray(handle["family_parameters"][:]).T
        external_metadata = {
            "N": int(handle["N"][0, 0]),
            "S": float(handle["S"][0, 0]),
            "L": float(handle["L"][0, 0]),
            "R": float(handle["R"][0, 0]),
            "Nrho": int(handle["Nrho"][0, 0]),
            "Nsph": int(handle["Nsph"][0, 0]),
            "Nsphpre": int(handle["Nsphpre"][0, 0]),
            "has_kernel_metadata": "kernel_name" in handle,
        }

    n = int(config["velocity"]["N"])
    n_per_family = int(config["external_test"]["samples_per_family"])
    n_external = 3 * n_per_family
    expected_input_shape = (n_external, 4, n, n, n)
    expected_target_shape = (n_external, 1, n, n, n)
    if physical_input.shape != expected_input_shape:
        raise ValueError(f"External input shape {physical_input.shape} != {expected_input_shape}")
    if target_physical.shape != expected_target_shape:
        raise ValueError(
            f"External target shape {target_physical.shape} != {expected_target_shape}"
        )
    if not np.isfinite(physical_input).all() or not np.isfinite(target_physical).all():
        raise ValueError("External data contain non-finite values")
    if family_parameters.shape != (n_external, 15):
        raise ValueError(f"Unexpected external parameter shape {family_parameters.shape}")
    unique_families, family_counts = np.unique(family_ids, return_counts=True)
    if not np.array_equal(unique_families, [1, 2, 3]) or not np.array_equal(
        family_counts, [n_per_family, n_per_family, n_per_family]
    ):
        raise ValueError("External family counts do not match config")
    if external_seed != int(config["external_test"]["seed"]):
        raise ValueError("External random seed does not match config")
    expected_numeric_metadata = {
        "N": n,
        "S": float(config["velocity"]["S"]),
        "L": float(config["velocity"]["L"]),
        "R": 2 * float(config["velocity"]["S"]),
        "Nrho": int(config["reference_solver"]["Nrho"]),
        "Nsph": int(config["reference_solver"]["Nsph"]),
        "Nsphpre": int(config["reference_solver"]["Nsphpre"]),
    }
    for name, expected in expected_numeric_metadata.items():
        actual = external_metadata[name]
        if not np.isclose(float(actual), float(expected)):
            raise ValueError(f"External metadata {name}={actual} != {expected}")
    if not external_metadata["has_kernel_metadata"]:
        raise ValueError("External data lack collision-kernel metadata")
    if not np.isclose(dv, float(config["velocity"]["dv"]), rtol=0, atol=1e-12):
        raise ValueError("External velocity spacing does not match config")
    if not np.isclose(dt, float(config["reference_solver"]["dt"])):
        raise ValueError("External time step does not match the reference solver")
    if not np.isclose(
        time_horizon, float(config["reference_solver"]["time_horizon"])
    ):
        raise ValueError("External map horizon does not match the trained endpoint map")
    expected_v = np.linspace(
        -float(config["velocity"]["L"]) + dv / 2,
        float(config["velocity"]["L"]) - dv / 2,
        n,
    )
    expected_coordinates = np.meshgrid(expected_v, expected_v, expected_v, indexing="ij")
    for axis, expected in enumerate(expected_coordinates):
        if not np.allclose(physical_input[:, axis], expected[None], rtol=0, atol=2e-6):
            raise ValueError(f"External velocity channel {axis} is inconsistent")
    velocity = tuple(physical_input[0, axis] for axis in range(3))
    initial_fields = physical_input[:, 3:4]
    rows: list[dict[str, Any]] = []

    predictions: dict[str, np.ndarray] = {
        "Identity": initial_fields.copy(),
        "Teacher": target_physical.copy(),
    }
    checkpoints = {}
    for kind, label in (("fno", "FNO"), ("cfno", "C-FNO")):
        checkpoint_path = project_dir / "checkpoints" / f"fno3d_t1_{kind}.pt"
        model, normalization, checkpoint = load_checkpoint(checkpoint_path, device)
        validate_checkpoint(checkpoint, kind, config, bundle)
        model_input = np.empty_like(physical_input)
        model_input[:, :3] = physical_input[:, :3] / normalization.velocity_scale
        model_input[:, 3:4] = (
            initial_fields - normalization.f_mean
        ) / normalization.f_std
        predictions[label] = predict_numpy(
            model,
            model_input,
            initial_fields,
            normalization,
            device,
        )
        checkpoints[label] = checkpoint_metadata(checkpoint_path, checkpoint)

    nearest_train: list[dict[str, Any]] = []
    family_counts_training = config["data"]["family_counts"]
    for sample_index, family in enumerate(family_ids):
        same_family_indices = [
            int(index)
            for index in bundle.splits["train"]
            if family_of(int(index), family_counts_training) == family
        ]
        distances = {
            index: relative_l2(
                bundle.input_physical[index, 3], physical_input[sample_index, 3]
            )
            for index in same_family_indices
        }
        nearest_index = min(distances, key=distances.get)
        nearest_train.append(
            {
                "sample_index_zero_based": sample_index,
                "family": int(family),
                "nearest_training_index_zero_based": nearest_index,
                "nearest_training_input_relative_L2": distances[nearest_index],
            }
        )

    for sample_index in range(len(family_ids)):
        reference = target_physical[sample_index, 0]
        initial = initial_fields[sample_index, 0]
        for method, fields in predictions.items():
            metrics = sample_metrics(
                fields[sample_index, 0], reference, initial, velocity, dv
            )
            rows.append(
                {
                    "sample_index_zero_based": int(sample_index),
                    "family": int(family_ids[sample_index]),
                    "method": method,
                    **metrics,
                }
            )

    csv_path = project_dir / "results" / "heldout_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": "six seed-separated external FSM trajectories generated after hyperparameters were frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_seed": external_seed,
        "external_data_sha256": file_sha256(external_path),
        "time_horizon": time_horizon,
        "dt": dt,
        "device": str(device),
        "checkpoints": checkpoints,
        "nearest_training_inputs": nearest_train,
        "overall": summarize_rows(rows, ("method",)),
        "by_family": summarize_rows(rows, ("method", "family")),
    }
    with (project_dir / "results" / "heldout_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved {csv_path}")


def load_bkw(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        input_data = np.transpose(handle["input_data"][:], (4, 3, 2, 1, 0)).astype(
            np.float32, copy=False
        )
        target_sm = np.transpose(handle["target_sm"][:], (4, 3, 2, 1, 0)).astype(
            np.float32, copy=False
        )
        target_true = np.transpose(handle["target_true"][:], (4, 3, 2, 1, 0)).astype(
            np.float32, copy=False
        )
        return {
            "input": input_data,
            "sm": target_sm,
            "true": target_true,
            "dv": float(handle["dv"][0, 0]),
            "v": handle["v"][:].reshape(-1),
            "physical_time_initial": float(handle["physical_time_initial"][0, 0]),
            "time_horizon": float(handle["time_horizon"][0, 0]),
            "dt": float(handle["dt"][0, 0]),
            "N": int(handle["N"][0, 0]),
            "L": float(handle["L"][0, 0]),
            "Nrho": int(handle["Nrho"][0, 0]),
            "Nsph": int(handle["Nsph"][0, 0]),
            "Nsphpre": int(handle["Nsphpre"][0, 0]),
        }


def interpolated_axis_profile(field: np.ndarray, axis: int) -> np.ndarray:
    """Linearly interpolate the two transverse coordinates to zero."""

    n = field.shape[0]
    if n % 2:
        selectors: list[Any] = [n // 2, n // 2, n // 2]
        selectors[axis] = slice(None)
        return field[tuple(selectors)]
    lower, upper = n // 2 - 1, n // 2
    transverse = [item for item in range(3) if item != axis]
    profiles = []
    for first in (lower, upper):
        for second in (lower, upper):
            selectors = [slice(None), slice(None), slice(None)]
            selectors[axis] = slice(None)
            selectors[transverse[0]] = first
            selectors[transverse[1]] = second
            profiles.append(field[tuple(selectors)])
    return np.mean(profiles, axis=0)


def interpolated_central_plane(field: np.ndarray, normal_axis: int = 2) -> np.ndarray:
    n = field.shape[normal_axis]
    if n % 2:
        return np.take(field, n // 2, axis=normal_axis)
    return 0.5 * (
        np.take(field, n // 2 - 1, axis=normal_axis)
        + np.take(field, n // 2, axis=normal_axis)
    )


def evaluate_bkw(project_dir: Path, config: dict[str, Any], device: torch.device) -> None:
    benchmark_path = project_dir / "data" / "bkw_benchmark_3d.mat"
    benchmark = load_bkw(benchmark_path)
    training_bundle = load_data_bundle(project_dir, config)
    expected_solver = config["reference_solver"]
    expected_bkw = config["bkw_benchmark"]
    checks = {
        "N": (benchmark["N"], int(config["velocity"]["N"])),
        "Nrho": (benchmark["Nrho"], int(expected_solver["Nrho"])),
        "Nsph": (benchmark["Nsph"], int(expected_solver["Nsph"])),
        "Nsphpre": (benchmark["Nsphpre"], int(expected_solver["Nsphpre"])),
    }
    if any(actual != expected for actual, expected in checks.values()):
        raise ValueError(f"Stale BKW integer metadata: {checks}")
    for name, actual, expected in (
        ("L", benchmark["L"], config["velocity"]["L"]),
        ("dt", benchmark["dt"], expected_solver["dt"]),
        (
            "physical_time_initial",
            benchmark["physical_time_initial"],
            expected_bkw["physical_time_initial"],
        ),
        ("time_horizon", benchmark["time_horizon"], expected_bkw["time_horizon"]),
        (
            "trained_map_time_horizon",
            benchmark["time_horizon"],
            expected_solver["time_horizon"],
        ),
    ):
        if not np.isclose(float(actual), float(expected)):
            raise ValueError(f"Stale BKW metadata for {name}: {actual} != {expected}")
    physical_input = benchmark["input"]
    initial = physical_input[:, 3:4]
    methods: dict[str, np.ndarray] = {
        "SM": benchmark["sm"][0, 0],
        "Identity": initial[0, 0],
    }

    checkpoints = {}
    for kind, label in (("fno", "FNO"), ("cfno", "C-FNO")):
        checkpoint_path = project_dir / "checkpoints" / f"fno3d_t1_{kind}.pt"
        model, normalization, checkpoint = load_checkpoint(checkpoint_path, device)
        validate_checkpoint(checkpoint, kind, config, training_bundle)
        model_input = np.empty_like(physical_input)
        model_input[:, :3] = physical_input[:, :3] / normalization.velocity_scale
        model_input[:, 3:4] = (initial - normalization.f_mean) / normalization.f_std
        methods[label] = predict_numpy(
            model, model_input, initial, normalization, device
        )[0, 0]
        checkpoints[label] = checkpoint_metadata(checkpoint_path, checkpoint)

    true = benchmark["true"][0, 0]
    velocity = tuple(physical_input[0, axis] for axis in range(3))
    dv = benchmark["dv"]
    mass_true, bulk_true, energy_true = numpy_moments(true, velocity, dv)
    rows: list[dict[str, Any]] = []
    for method in ("C-FNO", "FNO", "SM", "Identity"):
        field = methods[method]
        errors = distribution_errors(field, true, dv)
        mass, bulk, energy = numpy_moments(field, velocity, dv)
        rows.append(
            {
                "method": method,
                **errors,
                "relative_L2": relative_l2(field, true),
                "density_error": abs(mass - mass_true),
                "bulk_velocity_error": float(np.linalg.norm(bulk - bulk_true)),
                "energy_error": abs(energy - energy_true),
                "minimum": float(np.min(field)),
                "negative_mass": negative_mass(field, dv),
            }
        )

    csv_path = project_dir / "results" / "bkw_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    benchmark_initial = initial[0, 0]
    training_distances = {
        int(index): relative_l2(
            training_bundle.input_physical[index, 3], benchmark_initial
        )
        for index in training_bundle.splits["train"]
    }
    nearest_index = min(training_distances, key=training_distances.get)
    with (project_dir / "results" / "bkw_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "physical_time_initial": benchmark["physical_time_initial"],
                "time_convention": expected_bkw["time_convention"],
                "time_horizon": benchmark["time_horizon"],
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "device": str(device),
                "benchmark_sha256": file_sha256(benchmark_path),
                "checkpoints": checkpoints,
                "nearest_training_index_zero_based": nearest_index,
                "nearest_training_input_relative_L2": training_distances[nearest_index],
                "norms": "absolute full-volume midpoint-rule norms",
                "rows": rows,
            },
            handle,
            indent=2,
        )

    colors = {
        "True": "#1f77b4",
        "SM": "#ff7f0e",
        "C-FNO": "#2ca02c",
        "FNO": "#d62728",
        "Identity": "#6c757d",
    }
    styles = {"True": "-", "SM": ":", "C-FNO": "--", "FNO": "-.", "Identity": (0, (4, 2))}
    fields = {
        "True": true,
        **{name: methods[name] for name in ("SM", "C-FNO", "FNO", "Identity")},
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis_index, axis in enumerate(axes):
        for method, field in fields.items():
            axis.plot(
                benchmark["v"],
                interpolated_axis_profile(field, axis_index),
                linestyle=styles[method],
                color=colors[method],
                linewidth=2,
                label=method,
            )
        if axis_index == 0:
            axis.set_xlabel(r"$v_1$")
            axis.set_ylabel(r"$f(v_1,0,0)$")
            axis.set_title(r"$v_2=v_3=0$")
        else:
            axis.set_xlabel(r"$v_2$")
            axis.set_ylabel(r"$f(0,v_2,0)$")
            axis.set_title(r"$v_1=v_3=0$")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(r"Three-dimensional BKW benchmark: $f(5.5)\mapsto f(6.5)$")
    fig.savefig(project_dir / "figures" / "bkw_profiles_t1.png", dpi=250)
    fig.savefig(project_dir / "figures" / "bkw_profiles_t1.pdf")
    plt.close(fig)

    error_fno = np.abs(interpolated_central_plane(methods["FNO"] - true))
    error_cfno = np.abs(interpolated_central_plane(methods["C-FNO"] - true))
    maximum = max(float(error_fno.max()), float(error_cfno.max()))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    images = []
    for axis, error, title in zip(
        axes, (error_fno, error_cfno), ("FNO absolute error", "C-FNO absolute error")
    ):
        images.append(
            axis.imshow(
                error.T,
                origin="lower",
                extent=[benchmark["v"][0], benchmark["v"][-1]] * 2,
                vmin=0,
                vmax=maximum,
                cmap="magma",
                aspect="equal",
            )
        )
        axis.set(xlabel=r"$v_1$", ylabel=r"$v_2$", title=title + r" at $v_3=0$")
    fig.colorbar(images[-1], ax=axes, label="Absolute error", shrink=0.85)
    fig.savefig(project_dir / "figures" / "bkw_plane_errors_t1.png", dpi=250)
    fig.savefig(project_dir / "figures" / "bkw_plane_errors_t1.pdf")
    plt.close(fig)
    print(f"Saved {csv_path} and BKW figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir, config = load_config(args.config)
    device = choose_device(args.device or config["training"]["device"])
    evaluate_heldout(project_dir, config, device)
    evaluate_bkw(project_dir, config, device)


if __name__ == "__main__":
    main()
