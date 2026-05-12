# Reader Verification Guide

This repository is organized for three verification levels.

1. Release-level audit: run `python scripts/verify_release.py --root .` to check that expected source files, output tables, figures, and manifests are present.
2. Output-level audit: inspect `outputs/tables`, `outputs/figures/pdf`, and `results/db10_risk_boundary_audit_20260512` for the command-departure, context-prior, calibration, and user-decoder diagnostics.
3. Full rebuild: place raw public datasets outside the repository, set `DATA_ROOT`, and run the command in `README.md`.

The repository intentionally excludes raw datasets, posterior traces, model binaries, machine logs, and scheduler-specific paths. The compact results included here are sufficient to verify the reported aggregate tables and figure assets, but a full end-to-end rebuild requires separate lawful access to the original public datasets.
