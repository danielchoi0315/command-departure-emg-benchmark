# Command-Departure EMG Benchmark

This repository contains a minimal replication package for an open-data offline benchmark of decoder-relative command-departure policies for wearable myoelectric shared autonomy.

The release is intentionally limited to code, configuration, aggregate/unit-level outputs, audit tables, vector figures, and 1000-dpi figure exports. It does not redistribute raw participant data, raw electromyography arrays, video, posterior-trace archives, model checkpoints, machine-specific logs, or write-up-only material.

## Evidence Boundary

The supported claim is an offline sensor-informatics benchmark: calibrated user and assistive posterior traces are replayed through shared policies and evaluated with active macro-F1, risk-coverage area, intervention rate, and decoder-relative command-departure cost.

The package does not support online prosthetic control, patient-facing readiness, user-reported command ownership, embodiment, trust, workload, haptic feedback, or uniform policy dominance.

## Repository Layout

- `src/command_departure_benchmark/`: reusable benchmark code for feature handling, arbitration policies, metrics, splitting, calibration, and models.
- `scripts/`: preprocessing, training, simulation, audit, and release-verification entry points.
- `config/`: public-dataset registry and experiment configurations.
- `outputs/`: submitted tables, vector figure PDFs, 1000-dpi figure exports, and release manifests.
- `results/`: compact risk-boundary and context-control audit outputs used by the revised paper.
- `docs/`: dataset provenance and replication notes.

## Quick Verification

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py --root .
```

## Full Rebuild From Raw Public Data

Raw datasets must be obtained from their original repositories and placed outside version control. After accepting the relevant upstream dataset terms, set a local data root:

```bash
export DATA_ROOT=/path/to/public_emg_data
bash run_all.sh --tier 1 --exp command_departure_benchmark --paper_mode 1 --datasets_yaml config/datasets_command_departure.yaml --exp_yaml config/command_departure_suite.yaml
```

Hardware, filesystem paths, and scheduler settings are intentionally not hard-coded in the public package.

## Figure Assets

Submitted figure PDFs are in `outputs/figures/pdf`. Matching 1000-dpi PNG exports are in `outputs/figures/png_1000dpi`.

## Data Policy

Raw third-party datasets and trace-level posterior files are not redistributed. The repository provides code paths and aggregate outputs so that users with lawful access to the public datasets can rebuild the benchmark locally.
