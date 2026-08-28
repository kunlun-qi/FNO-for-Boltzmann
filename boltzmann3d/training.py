"""Deterministic training helpers for the refined 3D Boltzmann experiment."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn


def family_ids_from_counts(family_counts: list[int]) -> np.ndarray:
    """Return the one-based family label for the dataset's blocked ordering."""

    return np.concatenate(
        [np.full(count, family + 1, dtype=np.int64) for family, count in enumerate(family_counts)]
    )


def _single_item_stream(indices: Iterable[int], steps: int, seed: int) -> list[int]:
    """Cycle through independently shuffled permutations without dropping samples."""

    pool = np.asarray(list(indices), dtype=np.int64)
    if pool.size == 0:
        raise ValueError("A batch-schedule pool is empty")
    rng = np.random.default_rng(seed)
    queue: list[int] = []
    stream: list[int] = []
    for _ in range(steps):
        if not queue:
            queue.extend(int(index) for index in rng.permutation(pool))
        stream.append(queue.pop(0))
    return stream


def make_bkw_nonbkw_paired_schedule(
    train_indices: Iterable[int],
    family_ids: np.ndarray,
    steps: int,
    seed: int,
) -> list[list[int]]:
    """Pair one BKW and one non-BKW member in every optimizer update.

    This exactly implements the diagnostic sampler that repaired the BKW fit.
    The two pools are cycled independently, so exposure within each pool is
    uniform up to at most one sample.  It intentionally gives BKW half of the
    optimization exposure and is therefore a benchmark-prioritized sampler,
    not a claim that the underlying dataset itself has different proportions.
    """

    train = [int(index) for index in train_indices]
    bkw = [index for index in train if int(family_ids[index]) == 2]
    non_bkw = [index for index in train if int(family_ids[index]) != 2]
    left = _single_item_stream(bkw, steps, seed)
    right = _single_item_stream(non_bkw, steps, seed + 1)
    return [[bkw_index, other_index] for bkw_index, other_index in zip(left, right)]


def schedule_manifest(
    schedule: list[list[int]], family_ids: np.ndarray
) -> dict[str, object]:
    """Create a compact checksum and exact exposure counts for a schedule."""

    if not schedule:
        raise ValueError("The training schedule is empty")
    encoded = json.dumps(schedule, separators=(",", ":")).encode("utf-8")
    samples = Counter(index for batch in schedule for index in batch)
    families = Counter(int(family_ids[index]) for batch in schedule for index in batch)
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "optimizer_steps": len(schedule),
        "batch_size": len(schedule[0]),
        "sample_exposures": {str(key): value for key, value in sorted(samples.items())},
        "family_exposures": {str(key): value for key, value in sorted(families.items())},
    }


def initialize_small_random_final(
    model: nn.Module,
    *,
    seed: int,
    seed_offset: int,
    scale: float,
) -> None:
    """Reproduce the diagnostic winner's small final projection exactly.

    Lower layers retain the common CPU-created state.  Only the final 1x1x1
    convolution changes from zero to one fixed Kaiming-normal direction times
    ``scale``; its bias remains zero.  This lets gradients reach every Fourier
    block on the first backward pass.
    """

    if scale <= 0:
        raise ValueError("The final random scale must be positive")
    final = getattr(model, "projection", None)
    if final is None or not isinstance(final[-1], nn.Conv3d):
        raise TypeError("Expected model.projection[-1] to be a Conv3d")
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed + seed_offset)
    with torch.no_grad():
        nn.init.kaiming_normal_(final[-1].weight, a=math.sqrt(5.0))
        final[-1].weight.mul_(scale)
        nn.init.zeros_(final[-1].bias)
    torch.random.set_rng_state(rng_state)


def state_dict_sha256(model: nn.Module) -> str:
    """Hash a CPU model state, including real and complex parameters."""

    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def clip_gradient_norm(model: nn.Module, maximum: float) -> float:
    """Clip a joint real norm without relying on complex MPS operations."""

    squared_norm: torch.Tensor | None = None
    gradients: list[torch.Tensor] = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        gradients.append(gradient)
        contribution = gradient.real.square().sum()
        if torch.is_complex(gradient):
            contribution = contribution + gradient.imag.square().sum()
        squared_norm = contribution if squared_norm is None else squared_norm + contribution
    if squared_norm is None:
        return 0.0
    norm = squared_norm.sqrt()
    if maximum > 0:
        coefficient = (maximum / (norm + 1e-12)).clamp(max=1.0)
        for gradient in gradients:
            gradient.mul_(coefficient)
    return float(norm.detach().cpu())
