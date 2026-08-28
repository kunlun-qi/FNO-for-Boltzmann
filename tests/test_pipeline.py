"""Fast regression tests for tensor orientation, splitting, FNO, and moments."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from boltzmann3d.data import load_data_bundle  # noqa: E402
from boltzmann3d.models import FNO3d, SpectralConv3d  # noqa: E402
from boltzmann3d.physics import TorchVelocityGrid, torch_moments  # noqa: E402


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (PROJECT / "config.json").open("r", encoding="utf-8") as handle:
            cls.config = json.load(handle)
        cls.bundle = load_data_bundle(PROJECT, cls.config)

    def test_matlab_orientation_and_coordinates(self) -> None:
        n_samples = sum(self.config["data"]["family_counts"])
        n = self.config["velocity"]["N"]
        self.assertEqual(self.bundle.input_physical.shape, (n_samples, 4, n, n, n))
        self.assertEqual(self.bundle.target_physical.shape, (n_samples, 1, n, n, n))
        v1, v2, v3 = self.bundle.velocity
        self.assertGreater(float(np.ptp(v1[:, 0, 0])), 10.0)
        self.assertEqual(float(np.ptp(v1[0, :, 0])), 0.0)
        self.assertGreater(float(np.ptp(v2[0, :, 0])), 10.0)
        self.assertGreater(float(np.ptp(v3[0, 0, :])), 10.0)

    def test_stratified_splits_are_disjoint(self) -> None:
        splits = self.bundle.splits
        expected_sizes = {
            name: sum(self.config["data"]["split_counts"][name])
            for name in ("train", "validation", "test")
        }
        self.assertEqual({name: len(value) for name, value in splits.items()}, expected_sizes)
        merged = np.concatenate(list(splits.values()))
        n_samples = sum(self.config["data"]["family_counts"])
        self.assertEqual(len(np.unique(merged)), n_samples)
        self.assertEqual(set(merged.tolist()), set(range(n_samples)))

    def test_normalization_uses_training_subset(self) -> None:
        train = self.bundle.splits["train"]
        f0 = self.bundle.input_physical[train, 3:4]
        delta = self.bundle.target_physical[train] - f0
        self.assertAlmostEqual(self.bundle.normalization.f_mean, float(f0.mean(dtype=np.float64)))
        self.assertAlmostEqual(
            self.bundle.normalization.delta_std, float(delta.std(dtype=np.float64))
        )

    def test_zero_initialized_residual_and_resolution_compatibility(self) -> None:
        model = FNO3d(
            modes=(2, 2, 2),
            hidden_channels=4,
            num_layers=2,
            projection_channels=8,
        )
        for n in (8, 12):
            result = model(torch.randn(1, 4, n, n, n))
            self.assertEqual(result.shape, (1, 1, n, n, n))
            self.assertEqual(float(result.detach().abs().max()), 0.0)

    def test_spectral_layer_forward_and_complex_gradient(self) -> None:
        # The zero-initialized final FNO projection intentionally blocks a
        # whole-model spectral gradient at initialization, so exercise the
        # Fourier layer directly as a regression test.
        layer = SpectralConv3d(2, 3, modes=(2, 2, 2))
        x = torch.randn(2, 2, 8, 8, 8, requires_grad=True)
        output = layer(x)
        self.assertEqual(output.shape, (2, 3, 8, 8, 8))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for weight in layer.weights:
            self.assertIsNotNone(weight.grad)
            self.assertTrue(torch.isfinite(weight.grad.real).all())
            self.assertTrue(torch.isfinite(weight.grad.imag).all())
            self.assertGreater(float(weight.grad.abs().sum()), 0.0)

    def test_midpoint_moments(self) -> None:
        v1, v2, v3 = self.bundle.velocity
        grid = TorchVelocityGrid.from_numpy(self.bundle.velocity, self.bundle.dv, torch.device("cpu"))
        field = torch.from_numpy(self.bundle.input_physical[0:1, 3:4].copy())
        mass, momentum, energy, second = torch_moments(field, grid)
        self.assertLess(abs(float(mass) - 1.0), 2e-7)
        self.assertTrue(torch.isfinite(momentum).all())
        self.assertAlmostEqual(float(second), 2.0 * float(energy), places=6)
        self.assertEqual(v1.shape, v2.shape)
        self.assertEqual(v2.shape, v3.shape)


if __name__ == "__main__":
    unittest.main()
