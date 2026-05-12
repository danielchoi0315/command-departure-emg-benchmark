from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import (
    find_column,
    normalize_workload,
    one_hot_probs,
    read_table,
    uniform_probs,
    workload_to_classes,
    write_manifest,
    write_qc_report,
)

DATASET_ID = "gazeload"
WINDOW_MS = 250

_TIMESTAMP_COLS = ["timestamp", "time", "time_ms", "t_ms"]
_SUBJECT_COLS = ["subject_id", "subject", "participant", "participant_id", "id"]
_TRIAL_COLS = ["trial_id", "trial", "task", "task_id", "condition"]
_WORKLOAD_COLS = ["workload", "rating", "mental_workload", "nasa_tlx", "tlx"]
_ILLUM_COLS = ["illuminance", "lux", "ambient_light", "light_level"]
_PUPIL_COLS = ["pupil", "pupil_diameter", "pupil_left", "pupil_right", "pupil_mean"]


def _candidate_tables(raw_root: Path) -> list[Path]:
    exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}
    return sorted(path for path in raw_root.rglob("*") if path.is_file() and path.suffix.lower() in exts)


def _to_millis(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().sum() > 0:
        return num.ffill().fillna(0).astype(float)
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().sum() > 0:
        return (dt.astype("int64") / 1e6).astype(float)
    return pd.Series(np.arange(len(series), dtype=float) * WINDOW_MS, index=series.index)


def available(raw_root: Path) -> bool:
    # Mendeley package exposes participant folders P1/P2; tolerate generic table-only layout in tests.
    if (raw_root / "P1").exists() or (raw_root / "P2").exists():
        return True
    return len(_candidate_tables(raw_root)) > 0


def preprocess(raw_root: Path, out_root: Path, *, cfg: dict) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    tables = _candidate_tables(raw_root)
    if not tables:
        raise FileNotFoundError(f"No GAZELOAD tabular files found under {raw_root}")

    frames = []
    for table_path in tables:
        try:
            df = read_table(table_path)
        except Exception:
            continue
        if df.empty:
            continue
        cols = list(df.columns)
        ts_col = find_column(cols, _TIMESTAMP_COLS)
        wl_col = find_column(cols, _WORKLOAD_COLS)
        illum_col = find_column(cols, _ILLUM_COLS)
        if wl_col is None:
            continue

        pupil_cols = [col for col in cols if col.lower() in {cand.lower() for cand in _PUPIL_COLS}]
        if not pupil_cols:
            # Fallback: any column containing "pupil".
            pupil_cols = [col for col in cols if "pupil" in col.lower()]
        if not pupil_cols:
            continue

        subject_col = find_column(cols, _SUBJECT_COLS)
        trial_col = find_column(cols, _TRIAL_COLS)

        frame = pd.DataFrame(
            {
                "subject_id": df[subject_col].astype(str) if subject_col else table_path.parent.name,
                "trial_id": df[trial_col].astype(str) if trial_col else table_path.stem,
                "timestamp_ms": _to_millis(df[ts_col]) if ts_col else pd.Series(np.arange(len(df)) * WINDOW_MS),
                "workload_raw": pd.to_numeric(df[wl_col], errors="coerce"),
                "illuminance": pd.to_numeric(df[illum_col], errors="coerce") if illum_col else np.nan,
                "pupil_metric": df[pupil_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1),
                "source_file": table_path.relative_to(raw_root).as_posix(),
            }
        )
        frames.append(frame)

    if not frames:
        raise ValueError(
            "No parseable GAZELOAD rows found. Expected workload + pupil columns in at least one file."
        )

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df["bin_id"] = (raw_df["timestamp_ms"] // WINDOW_MS).astype(int)
    grouped = (
        raw_df.groupby(["subject_id", "trial_id", "bin_id"], as_index=False)
        .agg(
            workload_raw=("workload_raw", "mean"),
            illuminance=("illuminance", "mean"),
            pupil_metric=("pupil_metric", "mean"),
            source_file=("source_file", "first"),
        )
        .sort_values(["subject_id", "trial_id", "bin_id"], kind="stable")
    )
    grouped["workload"] = normalize_workload(grouped["workload_raw"])
    grouped["g_star"] = workload_to_classes(grouped["workload"], bins=3)

    n_classes = int(grouped["g_star"].max()) + 1
    rows = []
    for idx, row in grouped.reset_index(drop=True).iterrows():
        g_star = int(row["g_star"])
        rows.append(
            {
                "dataset": DATASET_ID,
                "subject_id": str(row["subject_id"]),
                "trial_id": str(row["trial_id"]),
                "window_id": int(idx),
                "window_bin_250ms": int(row["bin_id"]),
                "g_star": g_star,
                "p_u": one_hot_probs(g_star, n_classes=n_classes).tolist(),
                "p_a": uniform_probs(n_classes=n_classes).tolist(),
                "workload": float(row["workload"]),
                "pupil_metric": float(row["pupil_metric"]) if pd.notna(row["pupil_metric"]) else None,
                "illuminance": float(row["illuminance"]) if pd.notna(row["illuminance"]) else None,
                "raw_file": row["source_file"],
            }
        )

    out_df = pd.DataFrame(rows)
    windows_path = out_root / "windows.parquet"
    out_df.to_parquet(windows_path, index=False)
    qc = write_qc_report(
        out_root / "qc_report.json",
        dataset_id=DATASET_ID,
        df=out_df,
        extra={"window_ms": WINDOW_MS, "source_tables": [p.relative_to(raw_root).as_posix() for p in tables]},
    )
    write_manifest(
        out_root / "manifest.json",
        stage="adapter_preprocess",
        dataset_id=DATASET_ID,
        raw_root=raw_root,
        outputs=[windows_path, out_root / "qc_report.json"],
        cfg=cfg,
        qc=qc,
    )
    return windows_path
