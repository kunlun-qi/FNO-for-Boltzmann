"""Discrete 3D velocity moments and physically interpretable errors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TorchVelocityGrid:
    v1: torch.Tensor
    v2: torch.Tensor
    v3: torch.Tensor
    dv: float

    @classmethod
    def from_numpy(
        cls,
        velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
        dv: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "TorchVelocityGrid":
        tensors = [
            torch.as_tensor(component, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
            for component in velocity
        ]
        return cls(tensors[0], tensors[1], tensors[2], float(dv))


def torch_moments(
    f: torch.Tensor, grid: TorchVelocityGrid
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mass, momentum, physical energy, and the second moment.

    ``f`` must be ``[B,1,N1,N2,N3]``.  The second moment is retained because
    Eq. (3.13) in the paper penalizes ``int |v|^2 f dv`` without the factor 1/2.
    """

    if f.ndim != 5 or f.shape[1] != 1:
        raise ValueError(f"Expected [B,1,N1,N2,N3], got {f.shape}")
    volume = grid.dv**3
    reduce_dims = (-3, -2, -1)
    mass = f.sum(dim=reduce_dims).squeeze(1) * volume
    momentum = torch.stack(
        [
            (f * grid.v1).sum(dim=reduce_dims).squeeze(1),
            (f * grid.v2).sum(dim=reduce_dims).squeeze(1),
            (f * grid.v3).sum(dim=reduce_dims).squeeze(1),
        ],
        dim=1,
    ) * volume
    speed2 = grid.v1.square() + grid.v2.square() + grid.v3.square()
    second_moment = (f * speed2).sum(dim=reduce_dims).squeeze(1) * volume
    energy = 0.5 * second_moment
    return mass, momentum, energy, second_moment


def conservation_penalties(
    prediction: torch.Tensor, initial: torch.Tensor, grid: TorchVelocityGrid
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Squared mass, momentum, and energy-moment drift averaged over a batch."""

    mass_p, momentum_p, _, second_p = torch_moments(prediction, grid)
    mass_0, momentum_0, _, second_0 = torch_moments(initial, grid)
    mass_loss = (mass_p - mass_0).square().mean()
    momentum_loss = (momentum_p - momentum_0).square().sum(dim=1).mean()
    energy_loss = (second_p - second_0).square().mean()
    return mass_loss, momentum_loss, energy_loss


def scaled_conservation_penalties(
    prediction: torch.Tensor,
    initial: torch.Tensor,
    grid: TorchVelocityGrid,
    delta_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dimensionless form of the paper's three C-FNO penalties.

    The data loss is a *mean* squared error of the standardized increment,
    whereas Eq. (3.13) contains squared physical moments.  Adding those raw
    quantities with one coefficient gives mass, momentum, and energy very
    different effective weights.  For a collision invariant ``phi`` we use

    ``(int phi (prediction-initial) dv)^2 /
       (|D| * delta_std^2 * int phi^2 dv)``.

    This is still exactly a fixed-weight version of Eq. (3.13); the
    denominator only removes units and puts one invariant direction on the
    same scale as one direction in the standardized voxelwise MSE.  Moment
    reductions are performed in float64 on CPU to avoid cancellation in the
    nearly conservative residuals while the FNO itself remains float32.
    """

    if prediction.shape != initial.shape or prediction.ndim != 5:
        raise ValueError("prediction and initial must have the same [B,1,N,N,N] shape")
    if delta_std <= 0:
        raise ValueError("delta_std must be positive")

    delta = (prediction - initial).to(torch.float64)
    v1 = grid.v1.to(torch.float64)
    v2 = grid.v2.to(torch.float64)
    v3 = grid.v3.to(torch.float64)
    speed2 = v1.square() + v2.square() + v3.square()
    cell_volume = float(grid.dv) ** 3
    n_voxels = prediction.shape[-3] * prediction.shape[-2] * prediction.shape[-1]
    domain_volume = float(n_voxels) * cell_volume
    common = domain_volume * float(delta_std) ** 2
    reduce_dims = (-3, -2, -1)

    mass_drift = delta.sum(dim=reduce_dims).squeeze(1) * cell_volume
    momentum_drift = torch.stack(
        [
            (delta * component).sum(dim=reduce_dims).squeeze(1) * cell_volume
            for component in (v1, v2, v3)
        ],
        dim=1,
    )
    second_moment_drift = (
        (delta * speed2).sum(dim=reduce_dims).squeeze(1) * cell_volume
    )

    mass_denominator = common * domain_volume
    momentum_denominators = torch.stack(
        [common * component.square().sum() * cell_volume for component in (v1, v2, v3)]
    )
    energy_denominator = common * speed2.square().sum() * cell_volume

    mass_loss = (mass_drift.square() / mass_denominator).mean()
    momentum_loss = (
        momentum_drift.square() / momentum_denominators.reshape(1, 3)
    ).sum(dim=1).mean()
    energy_loss = (second_moment_drift.square() / energy_denominator).mean()
    return mass_loss, momentum_loss, energy_loss


def numpy_moments(
    f: np.ndarray,
    velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
    dv: float,
) -> tuple[float, np.ndarray, float]:
    """Mass, bulk velocity, and physical energy of one scalar 3D field."""

    field = np.asarray(f, dtype=np.float64)
    v1, v2, v3 = (np.asarray(item, dtype=np.float64) for item in velocity)
    volume = float(dv) ** 3
    mass = float(field.sum(dtype=np.float64) * volume)
    momentum = np.array(
        [
            np.sum(field * v1, dtype=np.float64),
            np.sum(field * v2, dtype=np.float64),
            np.sum(field * v3, dtype=np.float64),
        ]
    ) * volume
    bulk_velocity = momentum / mass
    energy = float(0.5 * np.sum(field * (v1**2 + v2**2 + v3**2)) * volume)
    return mass, bulk_velocity, energy


def distribution_errors(
    numerical: np.ndarray, reference: np.ndarray, dv: float
) -> dict[str, float]:
    """Absolute full-volume L1, L2, and Linfinity errors."""

    error = np.asarray(numerical, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    volume = float(dv) ** 3
    return {
        "L1": float(np.sum(np.abs(error), dtype=np.float64) * volume),
        "L2": float(np.sqrt(np.sum(error**2, dtype=np.float64) * volume)),
        "Linf": float(np.max(np.abs(error))),
    }


def relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    numerator = np.linalg.norm(np.asarray(numerical, dtype=np.float64).ravel() - np.asarray(reference, dtype=np.float64).ravel())
    denominator = np.linalg.norm(np.asarray(reference, dtype=np.float64).ravel())
    return float(numerator / denominator)
