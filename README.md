# 3D homogeneous Boltzmann equation with FNO and C-FNO

This repository-sized package reproduces the three-dimensional numerical
experiment for the spatially homogeneous Boltzmann equation with cutoff
Maxwell molecules, collision kernel `B = 1/(4*pi)`.  It contains:

- fast-spectral MATLAB data generation on a `32^3` velocity grid;
- the frozen 50-pair hybrid training dataset;
- FNO and conservative FNO (C-FNO) training and selected checkpoints;
- the exact positive BKW benchmark from physical time `t=5.5` to `t=6.5`;
- CPU and MPS inference timings and CPU fast-spectral timings;
- figures, machine-readable result tables, tests, and a PDF-report builder.

No LaTeX source is included.  The optional report is generated directly with
Python and is intentionally not tracked in this repository.

## Main numerical result

The reported models use eight Fourier modes per velocity axis, four Fourier
layers, width 16, and 50 endpoint pairs.  The exact BKW pair used below is not
in the training, validation, or checkpoint-selection sets.

| Method | Relative L2 error | Density error | Bulk-velocity error | Energy error |
|---|---:|---:|---:|---:|
| SM | 4.5981e-3 | 1.1939e-9 | 1.2517e-16 | 1.8704e-5 |
| FNO | 4.9332e-3 | 1.0602e-3 | 3.6495e-4 | 7.4118e-4 |
| C-FNO | 4.8018e-3 | 2.9350e-5 | 2.1609e-4 | 5.5386e-4 |

On the Apple M4 CPU used for the recorded experiment, one RK4 spectral step
took `1.7490 +/- 0.0221 s`, and the complete ten-step spectral map took
`17.5898 +/- 0.2530 s`.  Direct CPU endpoint inference took
`0.01429 +/- 0.00032 s` for FNO and `0.01440 +/- 0.00037 s` for C-FNO.
Training took 416.8 s and 417.3 s, respectively; training is an offline cost.

## Package layout

```text
boltzmann3d/       Python model, data, physics, and training modules
matlab/            Data generators and timing scripts
matlab/fast_spectral/
                   Collision solver and spherical-design data
data/              Frozen endpoint pairs, benchmark states, splits, manifest
checkpoints/       Selected FNO and C-FNO checkpoints
results/           CSV/JSON metrics, training histories, timing records
figures/           BKW profiles, plane errors, and training history
scripts/           Data-manifest and PDF-report utilities
tests/             Fast regression tests
```

The generated 480 MB fast-spectral weight cache is deliberately excluded from
the package and from Git.  MATLAB creates it automatically on first use.

## Python setup

Python 3.11 or newer is recommended.  The recorded run used PyTorch 2.8.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the tests and reproduce the benchmark figures and metrics from the shipped
checkpoints and data:

```bash
PYTHON_BIN=.venv/bin/python ./run_benchmark.sh
```

To repeat the spectral timing as well, provide MATLAB and opt in explicitly:

```bash
PYTHON_BIN=.venv/bin/python \
MATLAB_BIN=/Applications/MATLAB_R2025b.app/bin/matlab \
RUN_SPECTRAL_TIMING=1 ./run_benchmark.sh
```

Timing is hardware- and implementation-dependent.  The primary table compares
MATLAB double-precision SM and PyTorch float32 neural inference on the same
Apple M4 CPU; it is an implementation-level wall-clock comparison rather than
an equal-precision complexity result.

## Data generation

MATLAB R2025b was used for the frozen files.  From the package root:

```matlab
addpath('matlab');
generate_all_data_3d;
```

Generation proceeds in four auditable stages:

1. Create the deterministic baseline families.
2. Replace only the 17 BKW members with nine positive exact BKW states and
   eight bounded, mass-neutral perturbations near `f_BKW(5.5)`.
3. Generate the exact BKW benchmark and BKW time grid.
4. Generate a seed-separated six-pair external test set.

Every endpoint target uses RK4 with `dt=0.1` for one physical time unit.  After
intentional regeneration, validate the hybrid construction, refresh the
training-split normalization, and update the local data manifest before
training:

```bash
.venv/bin/python prepare_hybrid_data.py
.venv/bin/python scripts/make_data_manifest.py
```

The first generation run can be lengthy because it must construct and cache
the fast-spectral weights.  The cache is reusable but is not committed.

## Training

The package ships selected checkpoints, so training is optional.  To repeat
the deterministic CPU training:

```bash
PYTHON_BIN=.venv/bin/python ./run_training.sh
```

Both models use 3000 optimizer steps, small random final-layer initialization,
standardized endpoint increments, and BKW-balanced batches.  C-FNO adds
nondimensionalized mass, momentum, and energy penalties with weight 99.

## Rebuild the PDF report

```bash
.venv/bin/python scripts/build_report.py
```

This creates `report/boltzmann3d_numerical_report.pdf` locally.  The generated
`report/` directory is excluded from Git in this public repository.

## Reproducibility scope and limitations

- Results use one deterministic optimization seed.
- The exact BKW benchmark is evaluation-only.
- The models learn the fixed one-unit endpoint map, not arbitrary time steps.
- The spectral and neural implementations use different numerical precision.
- Wall-clock measurements will change with hardware and software versions.

## License

Original software and associated materials authored for this project are
released under the [MIT License](LICENSE), copyright 2026 Kunlun Qi and
contributors.  Externally sourced numerical components retain their
applicable original terms and attribution; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
