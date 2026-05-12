# Release Audit

This release contains only code, configuration, aggregate outputs, figures, and manifests required to inspect and reproduce the benchmark artifacts.

## Included

- Policy-layer benchmark code and configurations.
- Final aggregate tables and table sources.
- Vector PDF figures plus 1000-dpi PNG exports.
- Dataset provenance and artifact inventory manifests.
- Portable SHA-256 checksum manifest with POSIX-style relative paths.
- Verification script for release structure and figure resolution.

## Excluded

- Raw participant data and raw electromyography arrays.
- Video, MATLAB, HDF5, NumPy, WFDB, model-binary, archive, and checkpoint files.
- Local workstation paths, private compute-provider notes, and release-only materials.

## Data Boundary

Raw datasets must be obtained from their original providers under upstream terms. The release does not redistribute raw third-party datasets or trace-level posterior files.
