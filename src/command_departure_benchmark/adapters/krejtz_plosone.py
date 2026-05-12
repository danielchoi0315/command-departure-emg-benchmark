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
    write_manifest,
    write_qc_report,
)

DATASET_ID = "krejtz_plosone"

_SUBJECT_COLS = ["subject_id", "subject", "participant", "participant_id", "id", "subj"]
_TRIAL_COLS = ["trial_id", "trial", "condition", "task", "task_id", "block"]
_CONDITION_COLS = ["wm", "ttype", "ttypec", "ttypee", "condition", "difficulty", "ints"]
_WORKLOAD_COLS = ["nasa_tlx", "nasa_tlx_total", "workload", "mental_workload", "tlx", "wm_mean"]
_LABEL_COLS = ["g_star", "label", "class", "grasp_label"]
_PUPIL_COLS = ["pupil", "pupil_mean", "pupil_dilation", "pupil_diameter", "amp", "mag"]
_MICROSACCADE_COLS = ["microsaccade", "microsaccade_rate", "msrt", "pcpd", "bpcpd"]


def _candidate_tables(raw_root: Path) -> list[Path]:
    exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}
    return sorted(path for path in raw_root.rglob("*") if path.is_file() and path.suffix.lower() in exts)


def available(raw_root: Path) -> bool:
    for table_path in _candidate_tables(raw_root):
        try:
            df = read_table(table_path)
        except Exception:
            continue
        cols = list(df.columns)
        if find_column(cols, _SUBJECT_COLS) and find_column(cols, _WORKLOAD_COLS):
            return True
    return False


def _condition_to_binary_label(v: str) -> int | None:
    s = str(v).strip().lower()
    if not s:
        return None
    if s in {"low", "easy", "control", "baseline"}:
        return 0
    if s in {"high", "hard", "diff", "difficult"}:
        return 1
    if "easy" in s or "control" in s or "low" in s:
        return 0
    if "diff" in s or "hard" in s or "high" in s:
        return 1
    return None


def preprocess(raw_root: Path, out_root: Path, *, cfg: dict) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    role = str(cfg.get("arbitration_role", "workload_only")).strip().lower() or "workload_only"
    emit_proxy_probs = bool(cfg.get("emit_proxy_probs", True))
    tables = _candidate_tables(raw_root)
    if not tables:
        raise FileNotFoundError(f"No tabular files found under {raw_root}")

    parsed_frames = []
    used_tables: list[str] = []
    for table_path in tables:
        try:
            df = read_table(table_path)
        except Exception:
            continue
        if df.empty:
            continue
        cols = list(df.columns)
        workload_col = find_column(cols, _WORKLOAD_COLS)
        subject_col = find_column(cols, _SUBJECT_COLS)
        if workload_col is None or subject_col is None:
            continue
        trial_col = find_column(cols, _TRIAL_COLS)
        cond_col = find_column(cols, _CONDITION_COLS)
        label_col = find_column(cols, _LABEL_COLS)
        pupil_col = find_column(cols, _PUPIL_COLS)
        micro_col = find_column(cols, _MICROSACCADE_COLS)

        mini = pd.DataFrame(
            {
                "subject_id": df[subject_col].astype(str),
                "trial_id": (
                    df[trial_col].astype(str) if trial_col else pd.Series([f"trial_{i}" for i in range(len(df))])
                ),
                "condition_raw": df[cond_col].astype(str) if cond_col else pd.Series([""] * len(df)),
                "workload_raw": pd.to_numeric(df[workload_col], errors="coerce"),
                "g_star_raw": pd.to_numeric(df[label_col], errors="coerce") if label_col else pd.NA,
                "pupil_metric": pd.to_numeric(df[pupil_col], errors="coerce") if pupil_col else pd.NA,
                "microsaccade_metric": pd.to_numeric(df[micro_col], errors="coerce") if micro_col else pd.NA,
                "source_file": table_path.relative_to(raw_root).as_posix(),
            }
        )
        mini = mini.dropna(subset=["workload_raw"])
        parsed_frames.append(mini)
        used_tables.append(table_path.relative_to(raw_root).as_posix())

    if not parsed_frames:
        raise ValueError(
            "No parseable Krejtz tables found. Expected at least subject and workload columns."
        )

    merged = pd.concat(parsed_frames, ignore_index=True)
    merged["workload"] = normalize_workload(merged["workload_raw"])

    mapping_rule = "label_column"
    if merged["g_star_raw"].notna().any():
        labels = merged["g_star_raw"].fillna(merged["g_star_raw"].median()).astype(int)
        labels = labels - labels.min()
    else:
        cond_labels = merged["condition_raw"].map(_condition_to_binary_label)
        if cond_labels.notna().any():
            mapping_rule = "explicit_condition_mapping"
            labels = cond_labels.copy()
            # For unmapped condition strings, fallback to within-subject median split.
            for sid, grp in merged.groupby("subject_id", sort=False):
                mask = (merged["subject_id"] == sid) & labels.isna()
                if not mask.any():
                    continue
                med = float(np.nanmedian(grp["workload_raw"].to_numpy(dtype=float)))
                labels.loc[mask] = (merged.loc[mask, "workload_raw"] >= med).astype(int)
        else:
            mapping_rule = "within_subject_tlx_median_split"
            labels = pd.Series(index=merged.index, dtype=float)
            for sid, grp in merged.groupby("subject_id", sort=False):
                med = float(np.nanmedian(grp["workload_raw"].to_numpy(dtype=float)))
                idx = grp.index
                labels.loc[idx] = (grp["workload_raw"] >= med).astype(int)
        labels = labels.astype(int)
    merged["g_star"] = labels.astype(int)
    merged["workload_mapping_rule"] = mapping_rule

    n_classes = int(merged["g_star"].max()) + 1
    rows = []
    for idx, row in merged.reset_index(drop=True).iterrows():
        g_star = int(row["g_star"])
        pupil_metric = float(row["pupil_metric"]) if pd.notna(row["pupil_metric"]) else 0.0
        micro_metric = float(row["microsaccade_metric"]) if pd.notna(row["microsaccade_metric"]) else 0.0
        workload_cont = float(row["workload_raw"])
        out_row = {
            "dataset": DATASET_ID,
            "dataset_role": role,
            "subject_id": str(row["subject_id"]),
            "trial_id": str(row["trial_id"]),
            "window_id": int(idx),
            "g_star": g_star,
            "X_pu": [pupil_metric, micro_metric, workload_cont],
            "workload": float(row["workload"]),
            "workload_continuous": workload_cont,
            "workload_mapping_rule": str(row["workload_mapping_rule"]),
            "pupil_metric": float(row["pupil_metric"]) if pd.notna(row["pupil_metric"]) else None,
            "microsaccade_metric": (
                float(row["microsaccade_metric"]) if pd.notna(row["microsaccade_metric"]) else None
            ),
            "raw_condition": str(row["condition_raw"]),
            "raw_file": row["source_file"],
            "supports_pu_model": False,
        }
        if emit_proxy_probs:
            out_row["p_u"] = one_hot_probs(g_star, n_classes=n_classes).tolist()
            out_row["p_a"] = uniform_probs(n_classes=n_classes).tolist()
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    windows_path = out_root / "windows.parquet"
    out_df.to_parquet(windows_path, index=False)
    qc = write_qc_report(
        out_root / "qc_report.json",
        dataset_id=DATASET_ID,
        df=out_df,
        extra={
            "source_tables": used_tables,
            "workload_mapping_rules": [
                "label_column: use provided explicit labels when present",
                "explicit_condition_mapping: map wm/ttype condition strings to low/high",
                "within_subject_tlx_median_split: median split per subject on workload_raw",
            ],
            "dataset_role": role,
        },
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
