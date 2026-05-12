#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


POLICIES = {
    "UserOnly": "g_hat_useronly",
    "AutoOnly": "g_hat_autoonly",
    "ConfBlend": "g_hat_confblend",
    "SetACSA": "g_hat_setacsa",
    "RandomBudget": "g_hat_random_budget",
    "CSAAB": "g_hat_csaab",
    "CSAAB_BUDGET": "g_hat_csaab_budget",
}


def _metric_table(results_root: Path, exp_id: str) -> pd.DataFrame:
    rows = []
    for ds_dir in sorted((results_root / exp_id).glob("*")):
        sim_path = ds_dir / "sim.parquet"
        if not sim_path.exists():
            continue
        sim = pd.read_parquet(sim_path)
        if "subject_id" not in sim.columns:
            sim = sim.copy()
            sim["subject_id"] = "S0"
        for subject, sdf in sim.groupby("subject_id"):
            y = sdf["g_star"].astype(int).to_numpy()
            for policy, col in POLICIES.items():
                if col not in sdf.columns:
                    continue
                pred = sdf[col].astype(int).to_numpy()
                rows.append(
                    {
                        "dataset": ds_dir.name,
                        "subject_id": str(subject),
                        "policy": policy,
                        "accuracy": float(np.mean(y == pred)) if len(y) else float("nan"),
                        "macro_f1": float(f1_score(y, pred, average="macro")) if len(y) else float("nan"),
                    }
                )
    if not rows:
        raise FileNotFoundError(f"no sim.parquet files found for exp={exp_id}")
    df = pd.DataFrame(rows)
    return (
        df.groupby(["dataset", "policy"], as_index=False)
        .agg(accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"))
        .sort_values(["dataset", "policy"], kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare correctness and throughput command-departure benchmark policy summaries.")
    ap.add_argument("--exp_correctness", required=True)
    ap.add_argument("--exp_throughput", required=True)
    ap.add_argument("--results_root", type=Path, default=Path("results"))
    ap.add_argument("--tol_policy", type=float, default=0.001)
    ap.add_argument("--tol_meta", type=float, default=0.001)
    args = ap.parse_args()

    corr = _metric_table(args.results_root, args.exp_correctness)
    thr = _metric_table(args.results_root, args.exp_throughput)
    merged = corr.merge(thr, on=["dataset", "policy"], how="outer", suffixes=("_corr", "_thr"), indicator=True)
    failures = []
    max_diff = 0.0
    for _, row in merged.iterrows():
        if row["_merge"] != "both":
            failures.append({"dataset": row.get("dataset"), "policy": row.get("policy"), "reason": str(row["_merge"])})
            continue
        diffs = [
            abs(float(row["accuracy_mean_corr"]) - float(row["accuracy_mean_thr"])),
            abs(float(row["macro_f1_mean_corr"]) - float(row["macro_f1_mean_thr"])),
        ]
        max_diff = max(max_diff, max(diffs))
        if max(diffs) > float(args.tol_policy):
            failures.append(
                {
                    "dataset": row["dataset"],
                    "policy": row["policy"],
                    "accuracy_diff": diffs[0],
                    "macro_f1_diff": diffs[1],
                }
            )

    report = {
        "exp_correctness": args.exp_correctness,
        "exp_throughput": args.exp_throughput,
        "tol_policy": float(args.tol_policy),
        "max_policy_diff": float(max_diff),
        "rows_compared": int(len(merged)),
        "failures": failures,
    }
    out = args.results_root / args.exp_throughput / f"parity_vs_{args.exp_correctness}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
