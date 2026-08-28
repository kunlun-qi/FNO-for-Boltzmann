"""Regression tests for the repaired optimizer schedule and C-FNO scaling."""

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
from boltzmann3d.models import build_model  # noqa: E402
from boltzmann3d.physics import (  # noqa: E402
    TorchVelocityGrid,
    scaled_conservation_penalties,
)
from boltzmann3d.training import (  # noqa: E402
    family_ids_from_counts,
    initialize_small_random_final,
    make_bkw_nonbkw_paired_schedule,
    schedule_manifest,
    state_dict_sha256,
)


class RefinedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((PROJECT / "config.json").read_text(encoding="utf-8"))
        cls.bundle = load_data_bundle(PROJECT, cls.config)

    def test_paired_schedule_is_deterministic_and_balanced(self) -> None:
        family_ids = family_ids_from_counts(self.config["data"]["family_counts"])
        first = make_bkw_nonbkw_paired_schedule(
            self.bundle.splits["train"], family_ids, 101, 3727
        )
        second = make_bkw_nonbkw_paired_schedule(
            self.bundle.splits["train"], family_ids, 101, 3727
        )
        self.assertEqual(first, second)
        self.assertTrue(
            all(
                sorted(int(family_ids[index]) == 2 for index in batch) == [False, True]
                for batch in first
            )
        )
        manifest = schedule_manifest(first, family_ids)
        self.assertEqual(manifest["family_exposures"]["2"], 101)
        self.assertEqual(
            manifest["family_exposures"]["1"]
            + manifest["family_exposures"]["3"],
            101,
        )

    def test_small_random_final_is_nonzero_and_reproducible(self) -> None:
        hashes = []
        for _ in range(2):
            torch.manual_seed(int(self.config["seed"]))
            model = build_model(self.config["model"])
            initialize_small_random_final(
                model,
                seed=int(self.config["seed"]),
                seed_offset=991,
                scale=0.01,
            )
            hashes.append(state_dict_sha256(model))
            self.assertGreater(
                float(model.projection[-1].weight.detach().abs().max()), 0.0
            )
            self.assertEqual(float(model.projection[-1].bias.detach().abs().max()), 0.0)
        self.assertEqual(hashes[0], hashes[1])

    def test_scaled_penalty_matches_discrete_formula(self) -> None:
        grid = TorchVelocityGrid.from_numpy(
            self.bundle.velocity, self.bundle.dv, torch.device("cpu")
        )
        initial = torch.from_numpy(self.bundle.input_physical[0:1, 3:4].copy())
        rng = np.random.default_rng(91)
        delta_np = (1e-4 * rng.standard_normal(initial.shape)).astype(np.float32)
        prediction = initial + torch.from_numpy(delta_np)
        observed = scaled_conservation_penalties(
            prediction, initial, grid, self.bundle.normalization.delta_std
        )

        delta = (prediction - initial).double().numpy()[0, 0]
        velocity = [component.astype(np.float64) for component in self.bundle.velocity]
        speed2 = sum(component**2 for component in velocity)
        w = self.bundle.dv**3
        volume = delta.size * w
        sigma2 = self.bundle.normalization.delta_std**2
        mass = (delta.sum() * w) ** 2 / (volume * sigma2 * volume)
        momentum = sum(
            (np.sum(delta * component) * w) ** 2
            / (volume * sigma2 * np.sum(component**2) * w)
            for component in velocity
        )
        energy = (np.sum(delta * speed2) * w) ** 2 / (
            volume * sigma2 * np.sum(speed2**2) * w
        )
        for actual, expected in zip(observed, (mass, momentum, energy)):
            self.assertTrue(np.isclose(float(actual), expected, rtol=2e-11, atol=1e-14))


if __name__ == "__main__":
    unittest.main()
