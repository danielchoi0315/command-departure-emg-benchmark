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


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _as_prob_matrix(series: pd.Series) -> np.ndarray:
    mat = np.vstack(series.apply(lambda v: np.asarray(v, dtype=float).reshape(-1)))
    mat = np.clip(mat, 1e-12, None)
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def _id_keys(windows: pd.DataFrame, pred: pd.DataFrame) -> list[str]:
    preferred = ["dataset", "subject_id", "session_id", "trial_id", "window_id"]
    keys = [c for c in preferred if c in windows.columns and c in pred.columns]
    if not keys:
        raise ValueError("no shared id columns for pa_pred audit join")
    return keys


def _is_scalar_column(series: pd.Series) -> bool:
    for v in series.head(1000):
        if isinstance(v, (list, tuple, dict, np.ndarray)):
            return False
    return True


def _suspected_columns(df: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    skip = {
        "dataset",
        "subject_id",
        "session_id",
        "trial_id",
        "window_id",
        "window_idx_in_trial",
        "g_star",
        "p_u",
        "p_a",
        "p_u_pred",
        "p_a_pred",
        "X_pu",
        "emg_features",
        "raw_file",
        "fold_id",
        "method",
    }
    y = df["g_star"].astype(int)
    y_str = y.astype(str)

    rows: list[dict[str, Any]] = []
    for col in df.columns:
        if col in skip:
            continue
        s = df[col]
        if not _is_scalar_column(s):
            continue

        vals = s.fillna("__NA__").astype(str)
        eq_rate = float((vals == y_str).mean())

        grp = pd.DataFrame({"v": vals, "y": y})
        by_v = grp.groupby("v", observed=False)["y"].nunique()
        map_table = grp.groupby("v", observed=False)["y"].agg(lambda x: int(x.value_counts().idxmax()))
        pred_y = vals.map(map_table).astype(int)
        map_acc = float((pred_y == y).mean())

        by_y = grp.groupby("y", observed=False)["v"].nunique()
        one_to_one = bool((by_v <= 1).all() and (by_y <= 1).all())

        rows.append(
            {
                "column": col,
                "equal_rate": eq_rate,
                "deterministic_map_accuracy": map_acc,
                "one_to_one_mapping": one_to_one,
                "n_unique_values": int(vals.nunique()),
            }
        )

    rows.sort(key=lambda r: (r["deterministic_map_accuracy"], r["equal_rate"], r["n_unique_values"]), reverse=True)
    return rows[:top_k]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail-closed p_a oracle/leakage audit.")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--datasets_yaml", type=Path, default=Path("config/datasets.yaml"))
    ap.add_argument("--results_root", type=Path, default=Path("results"))
    ap.add_argument("--data_root", type=Path, default=Path(os.environ.get("DATA_ROOT", "lake")))
    ap.add_argument("--acc_threshold", type=float, default=0.98)
    ap.add_argument("--nll_threshold", type=float, default=1e-3)
    ap.add_argument("--top_k", type=int, default=8)
    args = ap.parse_args()

    dsreg = _load_yaml(args.datasets_yaml)["datasets"]
    exp_root = args.results_root / args.exp
    if not exp_root.exists():
        raise SystemExit(f"missing exp results root: {exp_root}")

    report_rows: list[dict[str, Any]] = []
    violations: list[str] = []

    for ds_id, info in dsreg.items():
        if int(info.get("tier", 99)) > args.tier:
            continue
        role = str(info.get("arbitration_role", "full")).strip().lower() or "full"
        if role != "full":
            continue

        windows_path = args.data_root / "derived" / ds_id / str(info.get("version", "unknown")) / "windows.parquet"
        if not windows_path.exists():
            continue

        pa_path = exp_root / ds_id / "PRED" / "pa_pred.parquet"
        if not pa_path.exists():
            violations.append(f"{ds_id}: missing pa_pred.parquet ({pa_path})")
            continue

        windows = pd.read_parquet(windows_path)
        pred = pd.read_parquet(pa_path)
        if "p_a_pred" not in pred.columns:
            violations.append(f"{ds_id}: pa_pred missing p_a_pred column")
            continue
        if pred.empty:
            violations.append(f"{ds_id}: pa_pred is empty")
            continue

        keys = _id_keys(windows, pred)
        if pred.duplicated(subset=keys).any():
            violations.append(f"{ds_id}: duplicate id rows in pa_pred")
            continue

        merged = pred.merge(windows, on=keys, how="left", suffixes=("", "_w"), validate="one_to_one")
        if "g_star" not in merged.columns and "g_star_w" in merged.columns:
            merged["g_star"] = merged["g_star_w"]
        if "g_star" not in merged.columns:
            violations.append(f"{ds_id}: unable to recover g_star for audit")
            continue
        if merged["g_star"].isna().any():
            violations.append(f"{ds_id}: missing g_star after pa_pred join")
            continue

        y = merged["g_star"].astype(int).to_numpy()
        probs = _as_prob_matrix(merged["p_a_pred"])
        pa_acc = float(np.mean(probs.argmax(axis=1) == y))
        pa_nll = _nll(probs, y)

        suspects = _suspected_columns(merged, top_k=args.top_k)
        report_rows.append(
            {
                "dataset": ds_id,
                "n_rows": int(len(merged)),
                "pa_accuracy": pa_acc,
                "pa_nll": pa_nll,
                "threshold_acc": float(args.acc_threshold),
                "threshold_nll": float(args.nll_threshold),
                "suspected_columns": suspects,
            }
        )

        print(f"[PA_ORACLE_AUDIT] {ds_id}: pa_accuracy={pa_acc:.6f} pa_nll={pa_nll:.6e}")
        if suspects:
            print(f"[PA_ORACLE_AUDIT] {ds_id}: top suspected leakage columns")
            for s in suspects[: min(5, len(suspects))]:
                print(
                    "  - "
                    f"{s['column']} eq_rate={s['equal_rate']:.4f} "
                    f"map_acc={s['deterministic_map_accuracy']:.4f} "
                    f"one_to_one={s['one_to_one_mapping']}"
                )

        if pa_acc > args.acc_threshold or pa_nll < args.nll_threshold:
            violations.append(
                f"{ds_id}: oracle-like p_a detected "
                f"(pa_accuracy={pa_acc:.6f}, pa_nll={pa_nll:.6e})"
            )

    report_path = exp_root / "pa_oracle_audit_report.json"
    report = {
        "exp": args.exp,
        "tier": int(args.tier),
        "results_root": str(exp_root),
        "acc_threshold": float(args.acc_threshold),
        "nll_threshold": float(args.nll_threshold),
        "datasets": report_rows,
        "violations": violations,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if violations:
        print("[PA_ORACLE_AUDIT] FAIL")
        for v in violations:
            print(f"- {v}")
        print(f"[PA_ORACLE_AUDIT] report: {report_path}")
        raise SystemExit(4)

    print(f"[PA_ORACLE_AUDIT] PASS (report: {report_path})")


if __name__ == "__main__":
    main()
