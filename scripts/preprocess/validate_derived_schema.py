#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _as_vector(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).reshape(-1)
    if pd.isna(value):
        return np.asarray([], dtype=float)
    return np.asarray([float(value)], dtype=float)


def _is_prob_vector(v: np.ndarray, atol: float = 1e-6) -> bool:
    if v.size == 0:
        return False
    if not np.isfinite(v).all():
        return False
    if np.any(v < -atol):
        return False
    s = float(v.sum())
    return abs(s - 1.0) <= 1e-3


def _validate_dataset(df: pd.DataFrame, *, dataset: str, role: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    required = ["g_star", "X_pu"]
    for col in required:
        if col not in df.columns:
            errors.append(f"{dataset}: missing required column {col}")

    n_rows = int(len(df))
    if "g_star" in df.columns:
        gs = pd.to_numeric(df["g_star"], errors="coerce")
        if gs.isna().any():
            errors.append(f"{dataset}: g_star contains NaN/non-numeric entries")

    xpu_dim_min = 0
    xpu_dim_max = 0
    xpu_nonempty = 0
    if "X_pu" in df.columns:
        dims: list[int] = []
        finite_ok = 0
        for raw in df["X_pu"]:
            vec = _as_vector(raw)
            dims.append(int(vec.size))
            if vec.size > 0:
                xpu_nonempty += 1
            if np.isfinite(vec).all():
                finite_ok += 1
        if dims:
            xpu_dim_min = int(min(dims))
            xpu_dim_max = int(max(dims))

        if role not in {"p_a_only", "workload_only"}:
            if xpu_nonempty == 0:
                errors.append(f"{dataset}: X_pu has no non-empty vectors")
            if xpu_dim_min <= 0:
                errors.append(f"{dataset}: X_pu contains empty vectors in arbitration-enabled dataset")
            if finite_ok != len(df):
                errors.append(f"{dataset}: X_pu contains non-finite values")
        else:
            # Role-gated datasets must still carry the column, but may use empty placeholders.
            if finite_ok != len(df):
                errors.append(f"{dataset}: X_pu contains non-finite values")

    debug_pu_valid = None
    if "p_u" in df.columns:
        ok = 0
        for raw in df["p_u"]:
            if _is_prob_vector(_as_vector(raw)):
                ok += 1
        debug_pu_valid = float(ok / max(1, len(df)))

    debug_pa_valid = None
    if "p_a" in df.columns:
        ok = 0
        for raw in df["p_a"]:
            if _is_prob_vector(_as_vector(raw)):
                ok += 1
        debug_pa_valid = float(ok / max(1, len(df)))

    summary = {
        "dataset": dataset,
        "dataset_role": role,
        "n_rows": n_rows,
        "has_g_star": bool("g_star" in df.columns),
        "has_X_pu": bool("X_pu" in df.columns),
        "x_pu_nonempty_rows": int(xpu_nonempty),
        "x_pu_dim_min": int(xpu_dim_min),
        "x_pu_dim_max": int(xpu_dim_max),
        "debug_p_u_valid_fraction": debug_pu_valid,
        "debug_p_a_valid_fraction": debug_pa_valid,
    }
    return summary, errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate required derived windows schema for end-to-end Command-departure benchmark.")
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--datasets_yaml", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, default=Path(os.environ.get("DATA_ROOT", "lake")))
    ap.add_argument("--out_report", type=Path, default=None)
    args = ap.parse_args()

    dsreg = load_yaml(args.datasets_yaml)["datasets"]
    derived_root = args.data_root / "derived"
    exp_id = os.environ.get("EXP_ID", "command_departure_template")
    out_report = args.out_report or (Path("results") / exp_id / "derived_schema_report.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    skipped_missing: list[str] = []

    for ds_id, info in dsreg.items():
        if int(info.get("tier", 99)) > args.tier:
            continue
        windows = derived_root / ds_id / info.get("version", "unknown") / "windows.parquet"
        if not windows.exists():
            skipped_missing.append(ds_id)
            continue
        try:
            df = pd.read_parquet(windows)
        except Exception as exc:
            failures.append(f"{ds_id}: failed reading {windows} ({exc})")
            continue
        role = str(info.get("arbitration_role", "full")).strip().lower() or "full"
        summary, errs = _validate_dataset(df, dataset=ds_id, role=role)
        summary["windows_path"] = str(windows)
        summaries.append(summary)
        failures.extend(errs)

    report = {
        "data_root": str(args.data_root),
        "derived_root": str(derived_root),
        "tier": int(args.tier),
        "n_validated": int(len(summaries)),
        "n_missing_windows": int(len(skipped_missing)),
        "missing_windows_datasets": skipped_missing,
        "n_failures": int(len(failures)),
        "failures": failures,
        "summaries": summaries,
    }
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if failures:
        print("[DERIVED_SCHEMA] FAIL")
        for err in failures:
            print(f"- {err}")
        print(f"[DERIVED_SCHEMA] report: {out_report}")
        raise SystemExit(1)

    print(f"[DERIVED_SCHEMA] PASS: validated {len(summaries)} datasets")
    if skipped_missing:
        print(f"[DERIVED_SCHEMA] skipped missing windows for datasets: {skipped_missing}")
    print(f"[DERIVED_SCHEMA] report: {out_report}")


if __name__ == "__main__":
    main()
