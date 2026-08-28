#!/usr/bin/env python3
"""Benchmark steady neural inference and combine it with MATLAB FSM timing."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from boltzmann3d.data import DataBundle, load_data_bundle
from boltzmann3d.runtime import (
    choose_device,
    file_sha256,
    load_checkpoint,
    load_config,
    reconstruct_distribution,
    synchronize,
)
from evaluate import load_bkw, validate_checkpoint


def timed_repetitions(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> np.ndarray:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        synchronize(device)

        elapsed = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            synchronize(device)
            started = time.perf_counter()
            function()
            synchronize(device)
            elapsed[index] = time.perf_counter() - started
    return elapsed


def statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_seconds": float(np.mean(values)),
        "std_seconds": float(np.std(values, ddof=1)),
        "median_seconds": float(np.median(values)),
        "minimum_seconds": float(np.min(values)),
    }


def spectral_rows(spectral: dict) -> list[dict]:
    """Normalize MATLAB JSON's scalar-struct versus struct-array encoding."""

    rows = spectral["rows"]
    return rows if isinstance(rows, list) else [rows]


def benchmark_model(
    project_dir: Path,
    kind: str,
    requested_device: str,
    physical_input: np.ndarray,
    warmup: int,
    repetitions: int,
    config: dict,
    bundle: DataBundle,
) -> dict:
    device = choose_device(requested_device)
    model, normalization, checkpoint = load_checkpoint(
        checkpoint_path := project_dir / "checkpoints" / f"fno3d_t1_{kind}.pt", device
    )
    validate_checkpoint(checkpoint, kind, config, bundle)
    initial_np = physical_input[:, 3:4].astype(np.float32, copy=False)
    model_input = np.empty_like(physical_input)
    model_input[:, :3] = physical_input[:, :3] / normalization.velocity_scale
    model_input[:, 3:4] = (initial_np - normalization.f_mean) / normalization.f_std
    input_cpu = torch.from_numpy(np.ascontiguousarray(model_input))
    initial_cpu = torch.from_numpy(np.ascontiguousarray(initial_np))
    input_device = input_cpu.to(device)
    initial_device = initial_cpu.to(device)

    def resident_call() -> torch.Tensor:
        return reconstruct_distribution(model(input_device), initial_device, normalization)

    def transfer_call() -> torch.Tensor:
        x = input_cpu.to(device)
        f0 = initial_cpu.to(device)
        return reconstruct_distribution(model(x), f0, normalization).to("cpu")

    resident = timed_repetitions(resident_call, device, warmup, repetitions)
    host_tensor_round_trip = timed_repetitions(
        transfer_call, device, warmup, repetitions
    )
    return {
        "method": kind.upper() if kind == "fno" else "C-FNO",
        "device": str(device),
        "N": int(physical_input.shape[-1]),
        "precision": "float32",
        "batch_size": 1,
        "warmup": warmup,
        "repetitions": repetitions,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "resident": statistics(resident),
        "host_tensor_round_trip": statistics(host_tensor_round_trip),
    }


def flatten_rows(neural_rows: list[dict], spectral: dict | None) -> list[dict]:
    rows: list[dict] = []
    if spectral is not None:
        for item in spectral_rows(spectral):
            rows.append(
                {
                    "method": "SM",
                    "device": "CPU",
                    "N": item["N"],
                    "precision": "double",
                    "batch_size": 1,
                    "mean_seconds": item["full_map_mean_seconds"],
                    "std_seconds": item["full_map_std_seconds"],
                    "timing_scope": "resident full RK4 f0-to-f1 map",
                    "warmup": item["warmup"],
                    "repetitions": item["repetitions"],
                }
            )
    for item in neural_rows:
        for scope in ("resident", "host_tensor_round_trip"):
            rows.append(
                {
                    "method": item["method"],
                    "device": item["device"],
                    "N": item["N"],
                    "precision": item["precision"],
                    "batch_size": item["batch_size"],
                    "mean_seconds": item[scope]["mean_seconds"],
                    "std_seconds": item[scope]["std_seconds"],
                    "timing_scope": scope,
                    "warmup": item["warmup"],
                    "repetitions": item["repetitions"],
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def cpu_brand() -> str:
    """Best-effort CPU identification for timing provenance."""

    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def main() -> None:
    args = parse_args()
    project_dir, config = load_config(args.config)
    benchmark = load_bkw(project_dir / "data" / "bkw_benchmark_3d.mat")
    bundle = load_data_bundle(project_dir, config)
    warmup = int(config["timing"]["neural_warmup"])
    repetitions = int(config["timing"]["neural_repetitions"])

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    elif torch.cuda.is_available():
        devices.insert(0, "cuda")
    neural_rows = [
        benchmark_model(
            project_dir,
            kind,
            device,
            benchmark["input"],
            warmup,
            repetitions,
            config,
            bundle,
        )
        for device in devices
        for kind in ("fno", "cfno")
    ]

    spectral_path = project_dir / "results" / "spectral_timing.json"
    spectral = None
    if spectral_path.exists():
        with spectral_path.open("r", encoding="utf-8") as handle:
            spectral = json.load(handle)

    speedups: dict[str, float] = {}
    speedups_by_device: dict[str, dict[str, float]] = {}
    break_even: dict[str, float] = {}
    if spectral is not None:
        sm_time = float(spectral_rows(spectral)[0]["full_map_mean_seconds"])
        training_summary_path = project_dir / "results" / "training_summary.json"
        training_seconds = {}
        if training_summary_path.exists():
            with training_summary_path.open("r", encoding="utf-8") as handle:
                training_summary = json.load(handle)
            training_seconds = {
                row["model"].upper() if row["model"] == "fno" else "C-FNO": row[
                    "training_seconds"
                ]
                for row in training_summary["runs"]
            }
        for row in neural_rows:
            online = row["host_tensor_round_trip"]["mean_seconds"]
            speedups_by_device.setdefault(row["device"], {})[row["method"]] = (
                sm_time / online
            )
            if row["device"] not in ("mps", "cuda"):
                continue
            speedups[row["method"]] = sm_time / online
            offline = float(config["data"]["generation_seconds"]) + float(
                training_seconds.get(row["method"], 0.0)
            )
            if sm_time > online:
                break_even[row["method"]] = offline / (sm_time - online)

    output = {
        "protocol": {
            "task": (
                "one complete BKW endpoint evaluation from physical time "
                f"{benchmark['physical_time_initial']:g} to "
                f"{benchmark['physical_time_initial'] + benchmark['time_horizon']:g}"
            ),
            "neural": (
                "batch one; inference mode; synchronized repetitions; float32; "
                "host-tensor round trip includes transfer, model, physical residual "
                "reconstruction, and output transfer, but excludes file I/O, MATLAB-axis "
                "conversion, and input normalization"
            ),
            "spectral": (
                "MATLAB CPU double RK4 with "
                f"{round(config['reference_solver']['time_horizon'] / config['reference_solver']['dt'])} "
                "steps; resident precomputed weights"
            ),
            "offline_training_excluded": True,
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_brand": cpu_brand(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
        "neural": neural_rows,
        "spectral": spectral,
        "online_speedup_sm_over_accelerated_host_tensor_round_trip": speedups,
        "online_speedup_sm_over_host_tensor_round_trip_by_device": speedups_by_device,
        "estimated_break_even_queries_including_data_and_training": break_even,
    }
    with (project_dir / "results" / "inference_timing.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(output, handle, indent=2)

    rows = flatten_rows(neural_rows, spectral)
    with (project_dir / "results" / "inference_timing.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
