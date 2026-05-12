from __future__ import annotations

import json
from pathlib import Path
import zipfile

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

DATASET_ID = "colet"

_SUBJECT_COLS = ["subject_id", "subject", "participant", "participant_id", "id", "subj"]
_TRIAL_COLS = ["trial_id", "trial", "task", "task_id", "condition", "block", "ttype"]
_WORKLOAD_COLS = ["workload", "nasa_tlx", "mental_workload", "tlx", "wm_mean"]
_LABEL_COLS = ["g_star", "label", "class"]
_PUPIL_COLS = ["pupil", "pupil_mean", "pupil_dilation", "pupil_diameter", "amp", "mag"]
_MICROSACCADE_COLS = ["microsaccade", "microsaccade_rate", "msrt", "pcpd", "bpcpd"]

_SUPPORTED_ZIPS = ("COLET_v3.zip", "COLET_v2.zip", "COLET_v1.zip")


def _zip_candidates(raw_root: Path) -> list[Path]:
    cands = []
    for name in _SUPPORTED_ZIPS:
        p = raw_root / name
        if p.exists():
            cands.append(p)
    # Accept any versioned zip following the same naming scheme.
    cands.extend(sorted(raw_root.glob("COLET_v*.zip")))
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _version_from_zip_name(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    lower = stem.lower()
    if lower.startswith("colet_v"):
        return lower.replace("colet_v", "v", 1)
    return lower


def _extract_zip(zip_path: Path, dest_root: Path) -> Path:
    version = _version_from_zip_name(zip_path.name)
    out_dir = dest_root / version
    marker = out_dir / ".extracted.ok"
    if marker.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    marker.write_text(f"source_zip={zip_path.name}\n", encoding="utf-8")
    return out_dir


def _materialize_extracted(raw_root: Path) -> list[Path]:
    extracted_root = raw_root / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)
    dirs: list[Path] = []
    for zip_path in _zip_candidates(raw_root):
        try:
            dirs.append(_extract_zip(zip_path, extracted_root))
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid COLET zip file: {zip_path} ({exc})") from exc
    # Include already extracted directories even if the zip is absent.
    dirs.extend(sorted(p for p in extracted_root.iterdir() if p.is_dir()))
    # Deduplicate
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in dirs:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _candidate_tables(roots: list[Path]) -> list[Path]:
    exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                if path in seen:
                    continue
                seen.add(path)
                out.append(path)
    return sorted(out)


def _mat_candidates(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        out.extend(sorted(p for p in root.rglob("*.mat") if p.is_file()))
    return out


def _looks_mcos_opaque(arr: np.ndarray) -> bool:
    if arr.dtype.kind not in {"u", "i"}:
        return False
    flat = np.asarray(arr).flatten()
    if flat.size != 6:
        return False
    # Typical unresolved MATLAB class signature observed in COLET v1-v3.
    # Example: [3707764736, 2, 1, 1, 13, 1]
    return int(flat[1]) == 2 and int(flat[2]) == 1 and int(flat[3]) == 1 and int(flat[5]) == 1


def _extract_numeric_metric(arr: np.ndarray) -> float | None:
    flat = pd.to_numeric(pd.Series(np.asarray(arr).flatten()), errors="coerce").dropna()
    if flat.empty:
        return None
    return float(flat.mean())


def _parse_tables(table_paths: list[Path], root_for_rel: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    used_tables: list[str] = []
    for table_path in table_paths:
        try:
            df = read_table(table_path)
        except Exception:
            continue
        if df.empty:
            continue
        cols = list(df.columns)
        workload_col = find_column(cols, _WORKLOAD_COLS)
        pupil_col = find_column(cols, _PUPIL_COLS)
        if workload_col is None or pupil_col is None:
            continue

        subject_col = find_column(cols, _SUBJECT_COLS)
        trial_col = find_column(cols, _TRIAL_COLS)
        label_col = find_column(cols, _LABEL_COLS)
        micro_col = find_column(cols, _MICROSACCADE_COLS)

        subject_vals = df[subject_col].astype(str) if subject_col else pd.Series(["S0"] * len(df))
        trial_vals = df[trial_col].astype(str) if trial_col else pd.Series([f"task_{i}" for i in range(len(df))])
        workload_vals = pd.to_numeric(df[workload_col], errors="coerce")
        pupil_vals = pd.to_numeric(df[pupil_col], errors="coerce")
        micro_vals = pd.to_numeric(df[micro_col], errors="coerce") if micro_col else pd.Series([pd.NA] * len(df))
        label_vals = pd.to_numeric(df[label_col], errors="coerce") if label_col else pd.Series([pd.NA] * len(df))

        for idx in range(len(df)):
            if pd.isna(workload_vals.iloc[idx]) or pd.isna(pupil_vals.iloc[idx]):
                continue
            rows.append(
                {
                    "subject_id": str(subject_vals.iloc[idx]),
                    "trial_id": str(trial_vals.iloc[idx]),
                    "workload_raw": float(workload_vals.iloc[idx]),
                    "pupil_metric": float(pupil_vals.iloc[idx]),
                    "microsaccade_metric": float(micro_vals.iloc[idx]) if pd.notna(micro_vals.iloc[idx]) else None,
                    "g_star_raw": float(label_vals.iloc[idx]) if pd.notna(label_vals.iloc[idx]) else None,
                    "source_file": table_path.relative_to(root_for_rel).as_posix(),
                    "workload_mapping_rule": "explicit_workload_column",
                }
            )
        used_tables.append(table_path.relative_to(root_for_rel).as_posix())
    return rows, used_tables


def _parse_mats(mat_paths: list[Path], root_for_rel: Path) -> tuple[list[dict], list[str], dict]:
    try:
        from pymatreader import read_mat
    except Exception as exc:
        raise RuntimeError(
            "COLET .mat parsing requires pymatreader. Install with `pip install pymatreader`."
        ) from exc

    rows: list[dict] = []
    used_mats: list[str] = []
    unresolved = 0
    unresolved_examples: list[list[int]] = []

    for mat_path in mat_paths:
        try:
            obj = read_mat(str(mat_path), variable_names=["Data"])
        except Exception:
            continue
        data = obj.get("Data")
        if not isinstance(data, dict):
            continue
        tasks = data.get("task")
        if not isinstance(tasks, list):
            continue

        for subj_idx, task_blob in enumerate(tasks):
            if not isinstance(task_blob, dict):
                continue
            ann_list = task_blob.get("annotation")
            pupil_list = task_blob.get("pupil")
            if not isinstance(ann_list, list) or not isinstance(pupil_list, list):
                continue
            n_tasks = min(len(ann_list), len(pupil_list))
            for task_idx in range(n_tasks):
                ann_arr = np.asarray(ann_list[task_idx])
                pupil_arr = np.asarray(pupil_list[task_idx])
                if _looks_mcos_opaque(ann_arr) or _looks_mcos_opaque(pupil_arr):
                    unresolved += 1
                    if len(unresolved_examples) < 3:
                        unresolved_examples.append([int(x) for x in np.asarray(ann_arr).flatten()[:6]])
                    continue
                workload_raw = _extract_numeric_metric(ann_arr)
                pupil_metric = _extract_numeric_metric(pupil_arr)
                if workload_raw is None or pupil_metric is None:
                    continue
                rows.append(
                    {
                        "subject_id": f"S{subj_idx + 1:02d}",
                        "trial_id": f"task_{task_idx + 1}",
                        "workload_raw": float(workload_raw),
                        "pupil_metric": float(pupil_metric),
                        "microsaccade_metric": None,
                        "g_star_raw": None,
                        "source_file": mat_path.relative_to(root_for_rel).as_posix(),
                        "workload_mapping_rule": "mat_annotation_mean_over_task",
                    }
                )
        used_mats.append(mat_path.relative_to(root_for_rel).as_posix())

    meta = {
        "mat_rows": len(rows),
        "mat_unresolved_mcos_rows": unresolved,
        "mat_unresolved_examples": unresolved_examples,
    }
    return rows, used_mats, meta


def available(raw_root: Path) -> bool:
    if _zip_candidates(raw_root):
        return True
    extracted = raw_root / "extracted"
    if extracted.exists() and any(p.is_file() for p in extracted.rglob("*.mat")):
        return True
    tables = _candidate_tables([raw_root])
    for table in tables:
        try:
            df = read_table(table)
        except Exception:
            continue
        cols = list(df.columns)
        if find_column(cols, _WORKLOAD_COLS) and find_column(cols, _PUPIL_COLS):
            return True
    return False


def preprocess(raw_root: Path, out_root: Path, *, cfg: dict) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    roots = [raw_root]
    roots.extend(_materialize_extracted(raw_root))

    table_paths = _candidate_tables(roots)
    table_rows, used_tables = _parse_tables(table_paths, raw_root)

    mat_paths = _mat_candidates(roots)
    mat_rows, used_mats, mat_meta = _parse_mats(mat_paths, raw_root)

    merged_rows = table_rows + mat_rows
    if not merged_rows:
        detail = {
            "n_tables_seen": len(table_paths),
            "n_mats_seen": len(mat_paths),
            **mat_meta,
        }
        raise ValueError(
            "No parseable COLET rows found. "
            "Expected workload + pupil signals from extracted COLET_v1/v2/v3 payloads. "
            f"details={json.dumps(detail)}"
        )

    merged = pd.DataFrame(merged_rows)
    merged["workload"] = normalize_workload(merged["workload_raw"])

    if merged["g_star_raw"].notna().any():
        labels = merged["g_star_raw"].fillna(merged["g_star_raw"].median()).astype(int)
        labels = labels - labels.min()
    else:
        labels = workload_to_classes(merged["workload"], bins=3)
    merged["g_star"] = labels.astype(int)

    n_classes = int(merged["g_star"].max()) + 1
    rows = []
    for idx, row in merged.reset_index(drop=True).iterrows():
        g_star = int(row["g_star"])
        rows.append(
            {
                "dataset": DATASET_ID,
                "subject_id": str(row["subject_id"]),
                "trial_id": str(row["trial_id"]),
                "window_id": int(idx),
                "g_star": g_star,
                "p_u": one_hot_probs(g_star, n_classes=n_classes).tolist(),
                "p_a": uniform_probs(n_classes=n_classes).tolist(),
                "workload": float(row["workload"]),
                "workload_continuous": float(row["workload_raw"]),
                "workload_mapping_rule": str(row["workload_mapping_rule"]),
                "pupil_metric": float(row["pupil_metric"]),
                "microsaccade_metric": (
                    float(row["microsaccade_metric"]) if pd.notna(row["microsaccade_metric"]) else None
                ),
                "raw_file": str(row["source_file"]),
            }
        )

    out_df = pd.DataFrame(rows)
    windows_path = out_root / "windows.parquet"
    out_df.to_parquet(windows_path, index=False)
    qc = write_qc_report(
        out_root / "qc_report.json",
        dataset_id=DATASET_ID,
        df=out_df,
        extra={
            "source_tables": used_tables,
            "source_mats": used_mats,
            "discovery": {
                "zip_candidates": [p.name for p in _zip_candidates(raw_root)],
                "table_candidates": [p.relative_to(raw_root).as_posix() for p in table_paths],
                "mat_candidates": [p.relative_to(raw_root).as_posix() for p in mat_paths],
                **mat_meta,
            },
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
