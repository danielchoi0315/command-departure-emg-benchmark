#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from command_departure_benchmark.eval.metrics import accuracy


POLICIES = {
    "UserOnly": "g_hat_useronly",
    "AutoOnly": "g_hat_autoonly",
    "ConfBlend": "g_hat_confblend",
    "SetACSA": "g_hat_setacsa",
    "RandomBudget": "g_hat_random_budget",
    "CSAAB": "g_hat_csaab",
    "CSAAB_Honly": "g_hat_csaab_entropy",
    "CSAAB_Wonly": "g_hat_csaab_workload",
}


def ci95(values: np.ndarray) -> tuple[float, float, float]:
    vals = values[~np.isnan(values)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if len(vals) < 2:
        return mean, mean, mean
    se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _flip_rate(pred: np.ndarray) -> float:
    if len(pred) < 2:
        return 0.0
    return float(np.mean(pred[1:] != pred[:-1]))


def subject_policy_metrics(sim: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if "H_pu" in sim.columns:
        unc_thr = float(sim["H_pu"].quantile(0.75))
    else:
        unc_thr = float("nan")
    if "workload" in sim.columns:
        w_thr = float(sim["workload"].quantile(0.75))
    else:
        w_thr = float("nan")

    rows = []
    for subject, sdf in sim.groupby("subject_id"):
        sdf = sdf.sort_values(["trial_id", "window_id"], kind="stable")
        y = sdf["g_star"].astype(int).to_numpy()
        high_unc = sdf["H_pu"].to_numpy() >= unc_thr if "H_pu" in sdf.columns and np.isfinite(unc_thr) else np.zeros(len(sdf), dtype=bool)
        high_w = sdf["workload"].to_numpy() >= w_thr if "workload" in sdf.columns and np.isfinite(w_thr) else np.zeros(len(sdf), dtype=bool)
        for policy_name, policy_col in POLICIES.items():
            if policy_col not in sdf.columns:
                continue
            pred = sdf[policy_col].astype(int).to_numpy()
            rows.append(
                {
                    "dataset": dataset,
                    "subject_id": subject,
                    "policy": policy_name,
                    "accuracy": accuracy(y, pred),
                    "macro_f1": float(f1_score(y, pred, average="macro")),
                    "override_rate": float(np.mean(pred != sdf["g_u"].astype(int).to_numpy())),
                    "flip_rate": _flip_rate(pred),
                    "acc_high_uncertainty": accuracy(y[high_unc], pred[high_unc]) if high_unc.any() else np.nan,
                    "acc_high_workload": accuracy(y[high_w], pred[high_w]) if high_w.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def aggregate_subject_metrics(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "accuracy",
        "macro_f1",
        "override_rate",
        "flip_rate",
        "acc_high_uncertainty",
        "acc_high_workload",
    ]
    for (dataset, policy), block in subject_df.groupby(["dataset", "policy"]):
        row: dict[str, Any] = {"dataset": dataset, "policy": policy, "n_subjects": int(block["subject_id"].nunique())}
        for metric in metric_cols:
            mean, lo, hi = ci95(block[metric].to_numpy(dtype=float))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_reference_exp_root(exp_root: Path, mode: str) -> tuple[Path, str]:
    jobs_root = exp_root / "jobs"
    ledger_path = exp_root / "RUN_LEDGER.csv"
    if jobs_root.exists() and ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        ledger = ledger[ledger["status"].isin(["completed", "skipped_existing"])].copy()
        if mode:
            ledger = ledger[ledger["mode"] == mode]
        if ledger.empty:
            raise SystemExit(f"no eligible ledger rows for mode={mode}")
        # Prefer completed rows for strict reproducibility reference.
        comp = ledger[ledger["status"] == "completed"].copy()
        base = comp if not comp.empty else ledger
        base = base.sort_values("timestamp_utc", kind="stable")
        row = base.iloc[-1]
        exp_id = str(row["exp_id"])
        return Path("results") / exp_id, exp_id
    return exp_root, exp_root.name


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute policy effects from sim.parquet and cross-check summary table.")
    ap.add_argument("--exp", required=True, help="Experiment id under results/ (supports suite roots)")
    ap.add_argument("--mode", default="correctness", help="Mode used to select latest suite job if exp is a suite root")
    ap.add_argument("--tolerance", type=float, default=1e-9)
    ap.add_argument("--table", type=Path, default=Path("paper/tables/policy_summary.csv"))
    args = ap.parse_args()

    exp_root = Path("results") / args.exp
    if not exp_root.exists():
        raise SystemExit(f"missing exp root: {exp_root}")

    ref_root, ref_id = resolve_reference_exp_root(exp_root, mode=args.mode)

    rows = []
    for sim_path in sorted(ref_root.glob("*/sim.parquet")):
        dataset = sim_path.parent.name
        sim = pd.read_parquet(sim_path)
        if "subject_id" not in sim.columns:
            sim["subject_id"] = "S0"
        if "trial_id" not in sim.columns:
            sim["trial_id"] = "T0"
        if "window_id" not in sim.columns:
            sim["window_id"] = np.arange(len(sim))
        rows.append(subject_policy_metrics(sim, dataset=dataset))

    if not rows:
        raise SystemExit(f"no sim.parquet files found under {ref_root}")

    subj_df = pd.concat(rows, ignore_index=True)
    recomputed = aggregate_subject_metrics(subj_df).sort_values(["dataset", "policy"], kind="stable").reset_index(drop=True)

    if not args.table.exists():
        raise SystemExit(f"missing summary table: {args.table}")
    table_df = pd.read_csv(args.table).sort_values(["dataset", "policy"], kind="stable").reset_index(drop=True)

    key_cols = ["accuracy_mean", "override_rate_mean", "flip_rate_mean"]
    merged = recomputed.merge(
        table_df[["dataset", "policy"] + key_cols],
        on=["dataset", "policy"],
        suffixes=("_recomputed", "_table"),
        how="outer",
        indicator=True,
    )

    failures: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        if row["_merge"] != "both":
            failures.append({
                "dataset": row.get("dataset"),
                "policy": row.get("policy"),
                "reason": f"row {row['_merge']}",
            })
            continue
        for c in key_cols:
            a = float(row[f"{c}_recomputed"])
            b = float(row[f"{c}_table"])
            diff = abs(a - b)
            if diff > args.tolerance:
                failures.append(
                    {
                        "dataset": row["dataset"],
                        "policy": row["policy"],
                        "metric": c,
                        "recomputed": a,
                        "table": b,
                        "abs_diff": diff,
                    }
                )

    out_report = {
        "exp": args.exp,
        "reference_exp_id": ref_id,
        "reference_root": str(ref_root),
        "table": str(args.table),
        "tolerance": float(args.tolerance),
        "n_rows_recomputed": int(len(recomputed)),
        "n_rows_table": int(len(table_df)),
        "n_failures": int(len(failures)),
        "failures": failures,
    }
    out_path = exp_root / "recompute_effects_check.json"
    out_path.write_text(json.dumps(out_report, indent=2), encoding="utf-8")

    if failures:
        print(f"[RECOMPUTE_CHECK] FAIL: {len(failures)} mismatches (see {out_path})")
        raise SystemExit(1)

    print(f"[RECOMPUTE_CHECK] PASS: exact match within tolerance={args.tolerance} (report: {out_path})")


if __name__ == "__main__":
    main()
