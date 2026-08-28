"""A reduced-width, resolution-compatible 3D Fourier neural operator."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv3d(nn.Module):
    """Low-mode Fourier convolution for a real 3D field.

    ``rfftn`` stores nonnegative frequencies in the last axis and both signs
    in the first two axes.  Four independent weight tensors cover those sign
    combinations, following the common FNO parameterization.  On the
    self-conjugate last-axis planes, unconstrained coefficients need not be
    Hermitian; ``irfftn`` returns their real projection.  Thus some boundary
    coefficients are redundant, but the output is always real.
    """

    def __init__(
        self, in_channels: int, out_channels: int, modes: Sequence[int]
    ) -> None:
        super().__init__()
        if len(modes) != 3 or any(int(mode) <= 0 for mode in modes):
            raise ValueError("modes must contain three positive integers")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = tuple(int(mode) for mode in modes)

        scale = 1.0 / (self.in_channels * self.out_channels) ** 0.5
        shape = (self.in_channels, self.out_channels, *self.modes)
        self.weights = nn.ParameterList(
            [nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat)) for _ in range(4)]
        )

    @staticmethod
    def _multiply(x_modes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", x_modes, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"SpectralConv3d expects [B,C,N1,N2,N3], got {x.shape}")
        n1, n2, n3 = x.shape[-3:]
        m1, m2, m3 = self.modes
        if m1 > n1 // 2 or m2 > n2 // 2 or m3 > n3 // 2 + 1:
            raise ValueError(f"Modes {self.modes} do not fit spatial shape {(n1, n2, n3)}")

        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            n1,
            n2,
            n3 // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device,
        )

        out_ft[:, :, :m1, :m2, :m3] = self._multiply(
            x_ft[:, :, :m1, :m2, :m3], self.weights[0]
        )
        out_ft[:, :, -m1:, :m2, :m3] = self._multiply(
            x_ft[:, :, -m1:, :m2, :m3], self.weights[1]
        )
        out_ft[:, :, :m1, -m2:, :m3] = self._multiply(
            x_ft[:, :, :m1, -m2:, :m3], self.weights[2]
        )
        out_ft[:, :, -m1:, -m2:, :m3] = self._multiply(
            x_ft[:, :, -m1:, -m2:, :m3], self.weights[3]
        )
        return torch.fft.irfftn(
            out_ft, s=(n1, n2, n3), dim=(-3, -2, -1), norm="ortho"
        )


class FourierBlock3d(nn.Module):
    """One global Fourier convolution plus a local channel-mixing map."""

    def __init__(self, width: int, modes: Sequence[int]) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes)
        self.local = nn.Conv3d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor, activate: bool = True) -> torch.Tensor:
        # No instance normalization is used here: removing a field's spatial
        # mean would obscure the zero Fourier mode that carries its mass.
        x = self.spectral(x) + self.local(x)
        return F.gelu(x) if activate else x


class FNO3d(nn.Module):
    """Point-to-point FNO used by both the MSE and conservative objectives.

    FNO and C-FNO deliberately share the exact architecture.  The distinction
    is only the training loss, matching the definition used in the paper.
    """

    def __init__(
        self,
        modes: Sequence[int] = (8, 8, 8),
        hidden_channels: int = 16,
        num_layers: int = 4,
        projection_channels: int = 32,
        in_channels: int = 4,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.config = {
            "modes": tuple(int(mode) for mode in modes),
            "hidden_channels": int(hidden_channels),
            "num_layers": int(num_layers),
            "projection_channels": int(projection_channels),
            "in_channels": int(in_channels),
            "out_channels": int(out_channels),
        }

        self.lifting = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FourierBlock3d(hidden_channels, modes) for _ in range(num_layers)]
        )
        self.projection = nn.Sequential(
            nn.Conv3d(hidden_channels, projection_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(projection_channels, out_channels, kernel_size=1),
        )
        # The learned quantity is f(1)-f(0).  A zero final projection makes
        # the untrained network exactly the strong identity-map baseline,
        # after which optimization only needs to learn the collision update.
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifting(x)
        for index, block in enumerate(self.blocks):
            x = block(x, activate=index + 1 < len(self.blocks))
        return self.projection(x)


def build_model(config: dict) -> FNO3d:
    """Construct a model from the ``model`` section of ``config.json``."""

    return FNO3d(
        modes=config["modes"],
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        projection_channels=config["projection_channels"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
    )
