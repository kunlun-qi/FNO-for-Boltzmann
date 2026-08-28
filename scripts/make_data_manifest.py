#!/usr/bin/env python3
"""Create the local SHA-256 manifest used before training."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from boltzmann3d.data import load_data_bundle


RELATIVE_FILES = (
    "data/input_data_3V.mat",
    "data/target_data_3V.mat",
    "data/input_data_3V_baseline50.mat",
    "data/target_data_3V_baseline50.mat",
    "data/fixed_split_indices.json",
    "data/bkw_benchmark_3d.mat",
    "data/bkw_time_grid_3d.mat",
    "data/external_test_data_3V.mat",
    "data/hybrid_bkw_metadata.mat",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [name for name in RELATIVE_FILES if not (PROJECT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot build manifest; missing: {missing}")
    manifest = {
        "manifest_kind": "immutable_training_and_evaluation_data",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {name: sha256(PROJECT / name) for name in RELATIVE_FILES},
    }
    output = PROJECT / "data" / "immutable_sha256.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    config = json.loads((PROJECT / "config.json").read_text(encoding="utf-8"))
    bundle = load_data_bundle(PROJECT, config)
    normalization_path = PROJECT / "data" / "normalization.npz"
    np.savez(normalization_path, **bundle.normalization.to_dict())
    print(f"Wrote {output} with {len(RELATIVE_FILES)} entries")
    print(f"Wrote training-split normalization to {normalization_path}")


if __name__ == "__main__":
    main()
