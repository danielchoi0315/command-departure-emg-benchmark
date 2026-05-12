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
    "CSAAB_Honly": "g_hat_csaab_entropy",
    "CSAAB_Wonly": "g_hat_csaab_workload",
}


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro"))


def _subject_policy_metrics(sim: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for subject, sdf in sim.groupby("subject_id"):
        y = sdf["g_star"].astype(int).to_numpy()
        g_u = sdf["g_u"].astype(int).to_numpy()
        for policy_name, col in POLICIES.items():
            if col not in sdf.columns:
                continue
            pred = sdf[col].astype(int).to_numpy()
            rows.append(
                {
                    "dataset": dataset,
                    "subject_id": str(subject),
                    "policy": policy_name,
                    "accuracy": _accuracy(y, pred),
                    "macro_f1": _macro_f1(y, pred),
                    "override_rate": float(np.mean(pred != g_u)),
                }
            )
    return pd.DataFrame(rows)


def _load_aggregate(results_root: Path, exp_id: str) -> pd.DataFrame:
    exp_dir = results_root / exp_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"missing exp dir: {exp_dir}")
    subject_rows = []
    for dataset_dir in sorted(exp_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        sim_path = dataset_dir / "sim.parquet"
        if not sim_path.exists():
            continue
        sim = pd.read_parquet(sim_path)
        if "g_star" not in sim.columns or "g_u" not in sim.columns:
            continue
        if "subject_id" not in sim.columns:
            sim["subject_id"] = "S0"
        subject_rows.append(_subject_policy_metrics(sim, dataset_dir.name))
    if not subject_rows:
        raise FileNotFoundError(f"no sim.parquet found for parity in {exp_dir}")
    subj_df = pd.concat(subject_rows, ignore_index=True)
    agg = (
        subj_df.groupby(["dataset", "policy"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            override_rate_mean=("override_rate", "mean"),
        )
        .sort_values(["dataset", "policy"], kind="stable")
    )
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Parity check for correctness vs throughput Command-departure benchmark Path-B runs.")
    ap.add_argument("--exp_correctness", required=True)
    ap.add_argument("--exp_throughput", required=True)
    ap.add_argument("--results_root", type=Path, default=Path("results"))
    ap.add_argument("--tol_accuracy", type=float, default=0.05)
    ap.add_argument("--tol_macro_f1", type=float, default=0.05)
    ap.add_argument("--tol_override", type=float, default=0.05)
    args = ap.parse_args()

    corr = _load_aggregate(args.results_root, args.exp_correctness)
    thr = _load_aggregate(args.results_root, args.exp_throughput)

    key = ["dataset", "policy"]
    merged = corr.merge(thr, on=key, how="inner", suffixes=("_corr", "_thr"))
    if merged.empty:
        raise SystemExit("[MODE_PARITY] FAIL: no common dataset/policy rows to compare")

    merged["accuracy_diff"] = (merged["accuracy_mean_corr"] - merged["accuracy_mean_thr"]).abs()
    merged["macro_f1_diff"] = (merged["macro_f1_mean_corr"] - merged["macro_f1_mean_thr"]).abs()
    merged["override_diff"] = (merged["override_rate_mean_corr"] - merged["override_rate_mean_thr"]).abs()

    bad = merged[
        (merged["accuracy_diff"] > float(args.tol_accuracy))
        | (merged["macro_f1_diff"] > float(args.tol_macro_f1))
        | (merged["override_diff"] > float(args.tol_override))
    ].copy()

    report = {
        "exp_correctness": args.exp_correctness,
        "exp_throughput": args.exp_throughput,
        "tol_accuracy": float(args.tol_accuracy),
        "tol_macro_f1": float(args.tol_macro_f1),
        "tol_override": float(args.tol_override),
        "n_compared": int(len(merged)),
        "n_failed": int(len(bad)),
        "max_accuracy_diff": float(merged["accuracy_diff"].max()),
        "max_macro_f1_diff": float(merged["macro_f1_diff"].max()),
        "max_override_diff": float(merged["override_diff"].max()),
        "failures": bad[
            [
                "dataset",
                "policy",
                "accuracy_diff",
                "macro_f1_diff",
                "override_diff",
            ]
        ].to_dict("records"),
    }

    out = args.results_root / args.exp_throughput / f"mode_parity_vs_{args.exp_correctness}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if len(bad) > 0:
        print("[MODE_PARITY] FAIL")
        for row in report["failures"][:20]:
            print(
                f"- {row['dataset']} {row['policy']}: "
                f"acc_diff={row['accuracy_diff']:.4f}, "
                f"f1_diff={row['macro_f1_diff']:.4f}, "
                f"override_diff={row['override_diff']:.4f}"
            )
        print(f"[MODE_PARITY] report: {out}")
        raise SystemExit(1)

    print(
        "[MODE_PARITY] PASS: "
        f"compared={len(merged)} "
        f"max_acc_diff={report['max_accuracy_diff']:.4f} "
        f"max_f1_diff={report['max_macro_f1_diff']:.4f} "
        f"max_override_diff={report['max_override_diff']:.4f} "
        f"report={out}"
    )


if __name__ == "__main__":
    main()
