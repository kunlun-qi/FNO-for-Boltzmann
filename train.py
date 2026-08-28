#!/usr/bin/env python3
"""Train matched refined FNO and C-FNO models on immutable hybrid data.

The diagnostics identified the MPS optimizer path as the dominant cause of
identity collapse.  This trainer therefore requires CPU, reproduces the
successful small-random final initialization and BKW/non-BKW paired batches,
and gives FNO and C-FNO exactly the same starting state and sample schedule.
C-FNO differs only by a dimensionless fixed-weight form of the paper's soft
mass, momentum, and second-moment penalties.  No data generator is called.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from boltzmann3d.data import DataBundle, load_data_bundle
from boltzmann3d.models import build_model
from boltzmann3d.physics import (
    TorchVelocityGrid,
    scaled_conservation_penalties,
    torch_moments,
)
from boltzmann3d.runtime import (
    file_sha256,
    load_config,
    model_parameter_count,
    model_real_scalar_parameter_count,
    reconstruct_distribution,
    set_reproducible_seed,
)
from boltzmann3d.training import (
    clip_gradient_norm,
    family_ids_from_counts,
    initialize_small_random_final,
    make_bkw_nonbkw_paired_schedule,
    schedule_manifest,
    state_dict_sha256,
)


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON atomically so an interruption cannot leave a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("Cannot save an empty training history")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clone_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def verify_immutable_inputs(
    project_dir: Path, config: dict[str, Any], bundle: DataBundle
) -> dict[str, str]:
    """Verify packaged tensors and splits against the local SHA-256 manifest."""

    manifest_path = project_dir / config["experiment"]["immutable_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run scripts/make_data_manifest.py after "
            "intentionally regenerating the data."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest.get("sha256", {})
    if not expected_hashes:
        raise ValueError(f"{manifest_path} contains no SHA-256 entries")

    hashes: dict[str, str] = {}
    for relative, expected_hash in expected_hashes.items():
        copied = project_dir / relative
        if not copied.exists():
            raise FileNotFoundError(f"Missing immutable artifact {relative}")
        copied_hash = file_sha256(copied)
        if copied_hash != expected_hash:
            raise RuntimeError(
                f"Packaged artifact {relative} changed: "
                f"{copied_hash} != {expected_hash}. Regenerate the manifest only "
                "when this data change is intentional."
            )
        hashes[relative] = copied_hash

    frozen_norm_path = project_dir / "data" / "normalization.npz"
    if frozen_norm_path.exists():
        with np.load(frozen_norm_path) as values:
            for name, actual in bundle.normalization.to_dict().items():
                expected = float(values[name])
                if not np.isclose(actual, expected, rtol=1e-7, atol=1e-14):
                    raise RuntimeError(
                        f"Training-only normalization changed for {name}: "
                        f"{actual} != {expected}"
                    )
    return hashes


def _batch_arrays(bundle: DataBundle, indices: list[int], device: torch.device):
    x = torch.from_numpy(np.ascontiguousarray(bundle.model_input[indices])).to(device)
    target = torch.from_numpy(np.ascontiguousarray(bundle.model_target[indices])).to(device)
    initial = torch.from_numpy(
        np.ascontiguousarray(bundle.input_physical[indices, 3:4])
    ).to(device)
    target_physical = torch.from_numpy(
        np.ascontiguousarray(bundle.target_physical[indices])
    ).to(device)
    return x, target, initial, target_physical


def objective_terms(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_delta: torch.Tensor,
    initial: torch.Tensor,
    bundle: DataBundle,
    velocity_grid: TorchVelocityGrid,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the matched FNO/C-FNO objective and auditable components."""

    predicted_delta = model(x)
    mse = F.mse_loss(predicted_delta, target_delta)
    prediction = reconstruct_distribution(predicted_delta, initial, bundle.normalization)
    mass, momentum, energy = scaled_conservation_penalties(
        prediction, initial, velocity_grid, bundle.normalization.delta_std
    )
    conservation = mass + momentum + energy
    loss = mse.to(torch.float64) + float(alpha) * conservation
    return loss, {
        "mse": mse,
        "mass": mass,
        "momentum": momentum,
        "energy": energy,
        "conservation": conservation,
    }


def evaluate_validation(
    model: torch.nn.Module,
    bundle: DataBundle,
    indices: list[int],
    family_ids: np.ndarray,
    velocity_grid: TorchVelocityGrid,
    device: torch.device,
    alpha: float,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate only the fixed validation split used for checkpoint selection."""

    totals = {
        name: 0.0
        for name in (
            "mse",
            "mass",
            "momentum",
            "energy",
            "mass_drift",
            "momentum_drift",
            "energy_drift",
            "negative_mass",
        )
    }
    endpoint_by_group: dict[str, list[float]] = {"bkw": [], "non_bkw": []}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            x, target_delta, initial, target_physical = _batch_arrays(
                bundle, batch, device
            )
            predicted_delta = model(x)
            prediction = reconstruct_distribution(
                predicted_delta, initial, bundle.normalization
            )
            batch_n = len(batch)
            mse = F.mse_loss(predicted_delta, target_delta)
            mass, momentum, energy = scaled_conservation_penalties(
                prediction, initial, velocity_grid, bundle.normalization.delta_std
            )
            for name, value in (
                ("mse", mse),
                ("mass", mass),
                ("momentum", momentum),
                ("energy", energy),
            ):
                totals[name] += float(value.cpu()) * batch_n

            endpoint_relative = (
                (prediction - target_physical).flatten(1).norm(dim=1)
                / target_physical.flatten(1).norm(dim=1).clamp_min(1e-12)
            )
            mass_p, momentum_p, energy_p, _ = torch_moments(prediction, velocity_grid)
            mass_0, momentum_0, energy_0, _ = torch_moments(initial, velocity_grid)
            mass_drift = (mass_p - mass_0).abs()
            momentum_drift = (momentum_p - momentum_0).norm(dim=1)
            energy_drift = (energy_p - energy_0).abs()
            negative_mass = prediction.clamp_max(0).abs().flatten(1).sum(dim=1) * (
                bundle.dv**3
            )
            totals["mass_drift"] += float(mass_drift.sum().cpu())
            totals["momentum_drift"] += float(momentum_drift.sum().cpu())
            totals["energy_drift"] += float(energy_drift.sum().cpu())
            totals["negative_mass"] += float(negative_mass.sum().cpu())
            for local, sample_index in enumerate(batch):
                group = "bkw" if int(family_ids[sample_index]) == 2 else "non_bkw"
                endpoint_by_group[group].append(float(endpoint_relative[local].cpu()))

    count = len(indices)
    if count == 0 or not endpoint_by_group["bkw"] or not endpoint_by_group["non_bkw"]:
        raise RuntimeError("Validation must contain BKW and non-BKW members")
    means = {name: value / count for name, value in totals.items()}
    bkw_relative = float(np.mean(endpoint_by_group["bkw"]))
    non_bkw_relative = float(np.mean(endpoint_by_group["non_bkw"]))
    relative = float(np.mean(endpoint_by_group["bkw"] + endpoint_by_group["non_bkw"]))
    selection_score = 0.5 * (bkw_relative + non_bkw_relative)
    return {
        "loss": means["mse"]
        + float(alpha) * (means["mass"] + means["momentum"] + means["energy"]),
        "mse": means["mse"],
        "mass": means["mass"],
        "momentum": means["momentum"],
        "energy": means["energy"],
        "relative_l2": relative,
        "bkw_relative_l2": bkw_relative,
        "non_bkw_relative_l2": non_bkw_relative,
        "selection_score": selection_score,
        "mass_drift": means["mass_drift"],
        "momentum_drift": means["momentum_drift"],
        "energy_drift": means["energy_drift"],
        "negative_mass": means["negative_mass"],
    }


def _resume_path(output_dir: Path, kind: str) -> Path:
    return output_dir / "checkpoints" / f".{kind}_resume.pt"


def save_resume(
    path: Path,
    *,
    kind: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_score: float,
    best_step: int,
    best_validation: dict[str, float],
    best_state: dict[str, torch.Tensor],
    history: list[dict[str, float]],
    elapsed: float,
    schedule_sha256: str,
    initial_state_sha256: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "kind": kind,
            "step": step,
            "state_dict": clone_cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": best_score,
            "best_step": best_step,
            "best_validation": best_validation,
            "best_state": best_state,
            "history": history,
            "elapsed": elapsed,
            "schedule_sha256": schedule_sha256,
            "initial_state_sha256": initial_state_sha256,
        },
        temporary,
    )
    temporary.replace(path)


def train_one(
    kind: str,
    config: dict[str, Any],
    bundle: DataBundle,
    output_dir: Path,
    device: torch.device,
    schedule: list[list[int]],
    schedule_info: dict[str, object],
    family_ids: np.ndarray,
    resume: bool,
) -> dict[str, Any]:
    """Train one model with fixed steps and validation-only checkpointing."""

    training_cfg = config["training"]
    seed = int(config["seed"])
    steps = int(training_cfg["optimizer_steps"])
    alpha = float(training_cfg["cfno_conservation_alpha"]) if kind == "cfno" else 0.0
    set_reproducible_seed(seed)
    model = build_model(config["model"]).cpu()
    initialize_small_random_final(
        model,
        seed=seed,
        seed_offset=int(training_cfg["final_random_seed_offset"]),
        scale=float(training_cfg["final_random_scale"]),
    )
    initial_state_sha256 = state_dict_sha256(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(training_cfg["minimum_learning_rate"]),
    )
    velocity_grid = TorchVelocityGrid.from_numpy(bundle.velocity, bundle.dv, device)
    configured_validation = config["data"].get("checkpoint_validation_indices")
    validation_indices = (
        [int(index) for index in configured_validation]
        if configured_validation is not None
        else [int(index) for index in bundle.splits["validation"]]
    )
    if not set(validation_indices).issubset(set(bundle.splits["validation"].tolist())):
        raise ValueError("checkpoint_validation_indices must be a validation subset")
    resume_path = _resume_path(output_dir, kind)
    start_step = 1
    elapsed_before = 0.0
    history: list[dict[str, float]] = []

    initial_validation = evaluate_validation(
        model,
        bundle,
        validation_indices,
        family_ids,
        velocity_grid,
        device,
        alpha,
        int(training_cfg["batch_size"]),
    )
    best_score = initial_validation["selection_score"]
    best_step = 0
    best_validation = dict(initial_validation)
    best_state = clone_cpu_state(model)

    if resume and resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved["kind"] != kind:
            raise RuntimeError(f"Stale resume model kind in {resume_path}")
        if saved["schedule_sha256"] != schedule_info["sha256"]:
            raise RuntimeError(f"Stale resume schedule in {resume_path}")
        if saved["initial_state_sha256"] != initial_state_sha256:
            raise RuntimeError(f"Stale resume initialization in {resume_path}")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_step = int(saved["step"]) + 1
        best_score = float(saved["best_score"])
        best_step = int(saved["best_step"])
        best_validation = dict(saved["best_validation"])
        best_state = saved["best_state"]
        history = list(saved["history"])
        elapsed_before = float(saved["elapsed"])
        print(f"Resuming {kind.upper()} at optimizer step {start_step}/{steps}")

    started = time.perf_counter()
    recent = {name: 0.0 for name in ("loss", "mse", "mass", "momentum", "energy")}
    recent_count = 0
    recent_gradient_norm = 0.0
    validation_every = int(training_cfg["validation_every_steps"])
    log_every = int(training_cfg["log_every_steps"])

    print(
        f"Training {kind.upper()} on CPU: {steps} steps, alpha={alpha:g}, "
        f"{model_parameter_count(model):,} tensor parameters"
    )
    for step in range(start_step, steps + 1):
        batch = schedule[step - 1]
        x, target_delta, initial, _ = _batch_arrays(bundle, batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, terms = objective_terms(
            model, x, target_delta, initial, bundle, velocity_grid, alpha
        )
        loss.backward()
        recent_gradient_norm = clip_gradient_norm(
            model, float(training_cfg["gradient_clip"])
        )
        optimizer.step()
        scheduler.step()
        recent["loss"] += float(loss.detach().cpu())
        for name in ("mse", "mass", "momentum", "energy"):
            recent[name] += float(terms[name].detach().cpu())
        recent_count += 1

        if step % validation_every == 0 or step == steps:
            validation = evaluate_validation(
                model,
                bundle,
                validation_indices,
                family_ids,
                velocity_grid,
                device,
                alpha,
                int(training_cfg["batch_size"]),
            )
            row = {
                "step": float(step),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": recent["loss"] / recent_count,
                "train_mse": recent["mse"] / recent_count,
                "train_mass": recent["mass"] / recent_count,
                "train_momentum": recent["momentum"] / recent_count,
                "train_energy": recent["energy"] / recent_count,
                "gradient_norm": recent_gradient_norm,
            }
            row.update({f"validation_{name}": value for name, value in validation.items()})
            history.append(row)
            recent = {name: 0.0 for name in recent}
            recent_count = 0

            score = validation["selection_score"]
            if score < best_score:
                best_score = score
                best_step = step
                best_validation = dict(validation)
                best_state = clone_cpu_state(model)

            elapsed = elapsed_before + time.perf_counter() - started
            if step % log_every == 0 or step == steps:
                print(
                    f"[{kind.upper()} {step:04d}/{steps}] "
                    f"val BKW={validation['bkw_relative_l2']:.3e}, "
                    f"non-BKW={validation['non_bkw_relative_l2']:.3e}, "
                    f"score={score:.3e}, best={best_score:.3e}@{best_step}"
                )
                save_resume(
                    resume_path,
                    kind=kind,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_score=best_score,
                    best_step=best_step,
                    best_validation=best_validation,
                    best_state=best_state,
                    history=history,
                    elapsed=elapsed,
                    schedule_sha256=str(schedule_info["sha256"]),
                    initial_state_sha256=initial_state_sha256,
                )

    elapsed = elapsed_before + time.perf_counter() - started
    final_state = clone_cpu_state(model)
    final_validation = evaluate_validation(
        model,
        bundle,
        validation_indices,
        family_ids,
        velocity_grid,
        device,
        alpha,
        int(training_cfg["batch_size"]),
    )
    checkpoint_common = {
        "format_version": 2,
        "model_kind": kind,
        "model_config": dict(config["model"]),
        "training_config": dict(training_cfg),
        "normalization": bundle.normalization.to_dict(),
        "splits": {name: values.tolist() for name, values in bundle.splits.items()},
        "checkpoint_validation_indices": validation_indices,
        "schedule": schedule_info,
        "initial_state_sha256": initial_state_sha256,
        "conservation_alpha": alpha,
        "checkpoint_selection": training_cfg["checkpoint_metric"],
    }
    best_checkpoint = {
        **checkpoint_common,
        "epoch": best_step,
        "optimizer_step": best_step,
        "validation_metrics": best_validation,
        "state_dict": best_state,
    }
    final_checkpoint = {
        **checkpoint_common,
        "epoch": steps,
        "optimizer_step": steps,
        "validation_metrics": final_validation,
        "state_dict": final_state,
    }
    checkpoint_path = output_dir / "checkpoints" / f"fno3d_t1_{kind}.pt"
    final_path = output_dir / "checkpoints" / f"fno3d_t1_{kind}_final.pt"
    torch.save(best_checkpoint, checkpoint_path)
    torch.save(final_checkpoint, final_path)
    history_path = output_dir / "results" / f"training_history_{kind}.csv"
    write_history(history_path, history)
    if resume_path.exists():
        resume_path.unlink()
    return {
        "model": kind,
        "checkpoint": str(checkpoint_path.relative_to(output_dir)),
        "final_checkpoint": str(final_path.relative_to(output_dir)),
        "history": str(history_path.relative_to(output_dir)),
        "parameters": model_parameter_count(model),
        "real_scalar_parameters": model_real_scalar_parameter_count(model),
        "optimizer_steps": steps,
        "training_seconds": elapsed,
        "best_step": best_step,
        "best_validation": best_validation,
        "final_validation": final_validation,
        "conservation_alpha": alpha,
        "initial_state_sha256": initial_state_sha256,
        "schedule_sha256": schedule_info["sha256"],
        "device": str(device),
    }


def plot_histories(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for summary in summaries:
        kind = summary["model"]
        history = np.genfromtxt(output_dir / summary["history"], delimiter=",", names=True)
        history = np.atleast_1d(history)
        axes[0].semilogy(
            history["step"], history["validation_bkw_relative_l2"], label=f"{kind.upper()} BKW"
        )
        axes[0].semilogy(
            history["step"],
            history["validation_non_bkw_relative_l2"],
            "--",
            label=f"{kind.upper()} non-BKW",
        )
        conservation = (
            history["validation_mass"]
            + history["validation_momentum"]
            + history["validation_energy"]
        )
        axes[1].semilogy(history["step"], conservation, label=kind.upper())
    axes[0].set(
        xlabel="Optimizer step",
        ylabel=r"Endpoint relative $L^2$ error",
        title="Common validation groups",
    )
    axes[1].set(
        xlabel="Optimizer step",
        ylabel="Dimensionless invariant loss",
        title="Validation conservation",
    )
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.savefig(output_dir / "figures" / "training_history.png", dpi=220)
    fig.savefig(output_dir / "figures" / "training_history.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", choices=("fno", "cfno", "both"), default="both")
    parser.add_argument("--device", default=None, help="Must be cpu for this repaired run")
    parser.add_argument("--steps", type=int, default=None, help="Smoke-test override")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Required with --steps so smoke outputs cannot overwrite final artifacts",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir, config = load_config(args.config)
    config = copy.deepcopy(config)
    if args.steps is not None:
        if args.steps <= 0 or args.output_root is None:
            raise ValueError("--steps requires a positive value and a separate --output-root")
        config["training"]["optimizer_steps"] = int(args.steps)
    requested_device = args.device or config["training"]["device"]
    device = torch.device(requested_device)
    if device.type != "cpu":
        raise RuntimeError(
            "The refined campaign must train on CPU; controlled diagnostics rejected MPS"
        )
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    output_dir = (
        (project_dir / args.output_root).resolve() if args.output_root else project_dir
    )
    for relative in ("checkpoints", "results", "figures"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)

    bundle = load_data_bundle(project_dir, config)
    immutable_hashes = verify_immutable_inputs(project_dir, config, bundle)
    family_ids = family_ids_from_counts(config["data"]["family_counts"])
    schedule = make_bkw_nonbkw_paired_schedule(
        bundle.splits["train"],
        family_ids,
        int(config["training"]["optimizer_steps"]),
        int(config["seed"]) + int(config["training"]["batch_seed_offset"]),
    )
    schedule_info = schedule_manifest(schedule, family_ids)
    atomic_json(
        output_dir / "results" / "training_schedule.json",
        {"batches": schedule, **schedule_info},
    )

    kinds = ["fno", "cfno"] if args.model == "both" else [args.model]
    summaries = [
        train_one(
            kind,
            config,
            bundle,
            output_dir,
            device,
            schedule,
            schedule_info,
            family_ids,
            not args.no_resume,
        )
        for kind in kinds
    ]
    if len(summaries) == 2 and (
        summaries[0]["initial_state_sha256"] != summaries[1]["initial_state_sha256"]
        or summaries[0]["schedule_sha256"] != summaries[1]["schedule_sha256"]
    ):
        raise RuntimeError("FNO and C-FNO did not receive identical initial states/schedules")
    plot_histories(output_dir, summaries)
    output = {
        "experiment": config["experiment"],
        "seed": int(config["seed"]),
        "split_sizes": {name: len(values) for name, values in bundle.splits.items()},
        "normalization": bundle.normalization.to_dict(),
        "immutable_input_sha256": immutable_hashes,
        "schedule": schedule_info,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "runs": summaries,
    }
    atomic_json(output_dir / "results" / "training_summary.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
