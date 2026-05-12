from __future__ import annotations
from pathlib import Path

DATASET_ID = "capgmyo"

def available(raw_root: Path) -> bool:
    # Detect if raw files are present
    return False

def preprocess(raw_root: Path, out_root: Path, *, cfg: dict) -> Path:
    """Convert raw dataset into standardized parquet windows.

    Requirements:
    - subject-wise identifiers stable
    - no leakage: do NOT compute statistics using test subjects
    - write a manifest.json with hashes and processing parameters
    """
    raise NotImplementedError("Implement dataset adapter for " + DATASET_ID)
