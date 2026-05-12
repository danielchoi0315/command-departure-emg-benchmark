#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def model_map(metrics: dict) -> dict[str, dict]:
    out = {}
    for row in metrics.get("models", []):
        model = row.get("model")
        if model:
            out[model] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare correctness vs throughput metrics for parity.")
    ap.add_argument("--exp_correctness", required=True)
    ap.add_argument("--exp_throughput", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--tol_accuracy", type=float, default=0.03)
    ap.add_argument("--tol_macro_f1", type=float, default=0.03)
    args = ap.parse_args()

    corr_root = Path("results") / args.exp_correctness
    thr_root = Path("results") / args.exp_throughput
    if not corr_root.exists() or not thr_root.exists():
        raise SystemExit("Missing experiment results for parity check.")

    datasets = []
    if args.dataset:
        datasets = [args.dataset]
    else:
        corr_ds = {p.name for p in corr_root.iterdir() if p.is_dir()}
        thr_ds = {p.name for p in thr_root.iterdir() if p.is_dir()}
        datasets = sorted(corr_ds & thr_ds)

    failures = []
    compared = 0
    for ds in datasets:
        corr_path = corr_root / ds / "metrics.json"
        thr_path = thr_root / ds / "metrics.json"
        if not corr_path.exists() or not thr_path.exists():
            continue
        corr = model_map(load_metrics(corr_path))
        thr = model_map(load_metrics(thr_path))
        for model in sorted(set(corr) & set(thr)):
            compared += 1
            a_corr = float(corr[model].get("accuracy_mean", 0.0))
            a_thr = float(thr[model].get("accuracy_mean", 0.0))
            f_corr = float(corr[model].get("macro_f1_mean", 0.0))
            f_thr = float(thr[model].get("macro_f1_mean", 0.0))
            if abs(a_corr - a_thr) > args.tol_accuracy or abs(f_corr - f_thr) > args.tol_macro_f1:
                failures.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "accuracy_diff": abs(a_corr - a_thr),
                        "macro_f1_diff": abs(f_corr - f_thr),
                    }
                )

    if compared == 0:
        print("[PARITY] No comparable dataset/model metrics found. Skipping parity assertion.")
        return

    if failures:
        print("[PARITY] FAIL: metric deltas exceeded tolerance")
        for item in failures:
            print(
                f"- {item['dataset']} {item['model']}: "
                f"accuracy_diff={item['accuracy_diff']:.4f}, macro_f1_diff={item['macro_f1_diff']:.4f}"
            )
        raise SystemExit(1)

    print(f"[PARITY] PASS: compared {compared} dataset/model pairs within tolerance.")


if __name__ == "__main__":
    main()
