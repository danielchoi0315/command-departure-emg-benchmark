#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze.tune_command_departure_policy_pareto import (
    _accuracy,
    _as_prob_matrix,
    _load_pred_vectors,
    _merge_prediction_features,
    _override,
    _subject_folds,
)


BASELINE_POLICIES = {
    "UserOnly": "g_hat_useronly",
    "AutoOnly": "g_hat_autoonly",
    "SetACSA": "g_hat_setacsa",
    "CSAAB": "g_hat_csaab",
    "CSAAB_BUDGET": "g_hat_csaab_budget",
    "RandomBudget": "g_hat_random_budget",
}

POLICY_LABELS = {
    "CSAAB": "CSAAB",
    "ProspectiveConfGateSetBudget": "ConfGate set-budget",
    "ProspectiveConfGateDevBudgetCap": "ConfGate dev-cap",
    "ProspectiveConfGatePareto": "ConfGate Pareto",
}


@dataclass(frozen=True)
class ExpInfo:
    exp_id: str
    base_exp: str
    mode: str
    model: str
    seed: int
    path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _discover(results_root: Path, base_exps: list[str], models: set[str] | None) -> list[ExpInfo]:
    out: list[ExpInfo] = []
    for base in base_exps:
        exp_re = re.compile(rf"^{re.escape(base)}_(?P<mode>correctness|throughput)_(?P<model>.+)_s(?P<seed>\d+)$")
        for p in sorted(results_root.glob(f"{base}_*")):
            if not p.is_dir():
                continue
            m = exp_re.match(p.name)
            if not m:
                continue
            model = m.group("model")
            if models and model not in models:
                continue
            out.append(
                ExpInfo(
                    exp_id=p.name,
                    base_exp=base,
                    mode=m.group("mode"),
                    model=model,
                    seed=int(m.group("seed")),
                    path=p,
                )
            )
    return out


def _threshold_for_target(score: np.ndarray, eligible: np.ndarray, *, target_override: float) -> float:
    if not eligible.any() or target_override <= 0:
        return float("inf")
    eligible_scores = np.asarray(score, dtype=float)[eligible]
    k = int(round(float(target_override) * len(score)))
    k = max(1, min(k, eligible_scores.size))
    ranked = np.sort(eligible_scores)[::-1]
    return float(ranked[k - 1])


def _apply(score: np.ndarray, g_u: np.ndarray, g_a: np.ndarray, threshold: float) -> np.ndarray:
    pred = np.asarray(g_u, dtype=int).copy()
    mask = (np.asarray(g_u, dtype=int) != np.asarray(g_a, dtype=int)) & (np.asarray(score, dtype=float) >= threshold)
    pred[mask] = np.asarray(g_a, dtype=int)[mask]
    return pred


def _candidate_thresholds(score: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    vals = np.asarray(score, dtype=float)[eligible]
    if vals.size == 0:
        return np.asarray([float("inf")], dtype=float)
    qs = np.linspace(0.0, 1.0, 101)
    try:
        thr = np.quantile(vals, qs, method="linear")
    except TypeError:
        thr = np.quantile(vals, qs, interpolation="linear")
    thr = np.unique(np.concatenate([[float("inf")], thr, [np.nextafter(vals.min(), -np.inf), np.nextafter(vals.max(), np.inf)]]))
    return np.sort(thr)[::-1]


def _select_best_threshold(
    *,
    y_dev: np.ndarray,
    score_dev: np.ndarray,
    g_u_dev: np.ndarray,
    g_a_dev: np.ndarray,
    max_override: float | None,
) -> tuple[float, dict[str, float]]:
    eligible = g_u_dev != g_a_dev
    best_key: tuple[float, float] | None = None
    best_thr = float("inf")
    best_metrics = {"dev_accuracy": _accuracy(y_dev, g_u_dev), "dev_override": 0.0}
    for thr in _candidate_thresholds(score_dev, eligible):
        pred = _apply(score_dev, g_u_dev, g_a_dev, float(thr))
        ov = _override(pred, g_u_dev)
        if max_override is not None and ov > max_override + 1e-12:
            continue
        acc = _accuracy(y_dev, pred)
        key = (acc, -ov)
        if best_key is None or key > best_key:
            best_key = key
            best_thr = float(thr)
            best_metrics = {"dev_accuracy": acc, "dev_override": ov}
    best_metrics["threshold"] = best_thr
    return best_thr, best_metrics


def _row(
    *,
    exp: ExpInfo,
    dataset: str,
    split_id: int,
    policy: str,
    y: np.ndarray,
    g_u: np.ndarray,
    pred: np.ndarray,
    set_acc: float,
    set_override: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    acc = _accuracy(y, pred)
    ov = _override(pred, g_u)
    out: dict[str, Any] = {
        "base_exp": exp.base_exp,
        "exp_id": exp.exp_id,
        "model": exp.model,
        "mode": exp.mode,
        "seed": int(exp.seed),
        "dataset": dataset,
        "split_id": int(split_id),
        "policy": policy,
        "accuracy": acc,
        "override_rate": ov,
        "delta_accuracy_vs_setacsa": acc - set_acc,
        "delta_override_vs_setacsa": ov - set_override,
        "n_windows": int(len(y)),
    }
    if extra:
        out.update(extra)
    return out


def _process(exp: ExpInfo, dataset: str, *, n_splits: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sim_path = exp.path / dataset / "sim.parquet"
    if not sim_path.exists():
        return [], [{"exp_id": exp.exp_id, "dataset": dataset, "warning": f"missing {sim_path}"}]
    sim = pd.read_parquet(sim_path)
    pu, pa = _load_pred_vectors(exp.path, dataset)
    df = _merge_prediction_features(sim, pu, pa)
    p_a = _as_prob_matrix(df["p_a_pred"])
    c_a = p_a.max(axis=1)
    y = df["g_star"].to_numpy(dtype=int)
    g_u = df["g_u"].to_numpy(dtype=int)
    g_a = df["g_a"].to_numpy(dtype=int)
    subjects = df["subject_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    folds = _subject_folds(subjects, n_splits=n_splits, seed=exp.seed + sum(ord(ch) for ch in dataset) + 7919)
    for split_id, (dev_idx, test_idx) in enumerate(folds):
        y_dev, y_test = y[dev_idx], y[test_idx]
        gu_dev, gu_test = g_u[dev_idx], g_u[test_idx]
        ga_dev, ga_test = g_a[dev_idx], g_a[test_idx]
        score_dev, score_test = c_a[dev_idx], c_a[test_idx]
        set_dev = df["g_hat_setacsa"].to_numpy(dtype=int)[dev_idx]
        set_test = df["g_hat_setacsa"].to_numpy(dtype=int)[test_idx]
        set_acc = _accuracy(y_test, set_test)
        set_ov_dev = _override(set_dev, gu_dev)
        set_ov_test = _override(set_test, gu_test)

        for policy, col in BASELINE_POLICIES.items():
            if col in df.columns:
                rows.append(
                    _row(
                        exp=exp,
                        dataset=dataset,
                        split_id=split_id,
                        policy=policy,
                        y=y_test,
                        g_u=gu_test,
                        pred=df[col].to_numpy(dtype=int)[test_idx],
                        set_acc=set_acc,
                        set_override=set_ov_test,
                    )
                )

        thr = _threshold_for_target(score_dev, gu_dev != ga_dev, target_override=set_ov_dev)
        pred = _apply(score_test, gu_test, ga_test, thr)
        rows.append(
            _row(
                exp=exp,
                dataset=dataset,
                split_id=split_id,
                policy="ProspectiveConfGateSetBudget",
                y=y_test,
                g_u=gu_test,
                pred=pred,
                set_acc=set_acc,
                set_override=set_ov_test,
                extra={"threshold": thr, "dev_target_override": set_ov_dev},
            )
        )

        thr, chosen = _select_best_threshold(
            y_dev=y_dev,
            score_dev=score_dev,
            g_u_dev=gu_dev,
            g_a_dev=ga_dev,
            max_override=set_ov_dev,
        )
        pred = _apply(score_test, gu_test, ga_test, thr)
        rows.append(
            _row(
                exp=exp,
                dataset=dataset,
                split_id=split_id,
                policy="ProspectiveConfGateDevBudgetCap",
                y=y_test,
                g_u=gu_test,
                pred=pred,
                set_acc=set_acc,
                set_override=set_ov_test,
                extra=chosen,
            )
        )

        thr, chosen = _select_best_threshold(
            y_dev=y_dev,
            score_dev=score_dev,
            g_u_dev=gu_dev,
            g_a_dev=ga_dev,
            max_override=None,
        )
        pred = _apply(score_test, gu_test, ga_test, thr)
        rows.append(
            _row(
                exp=exp,
                dataset=dataset,
                split_id=split_id,
                policy="ProspectiveConfGatePareto",
                y=y_test,
                g_u=gu_test,
                pred=pred,
                set_acc=set_acc,
                set_override=set_ov_test,
                extra=chosen,
            )
        )
        diagnostics.append(
            {
                "exp_id": exp.exp_id,
                "dataset": dataset,
                "split_id": int(split_id),
                "setacsa_dev_override": set_ov_dev,
                "setacsa_test_override": set_ov_test,
                "c_a_dev_min": float(np.min(score_dev)),
                "c_a_dev_median": float(np.median(score_dev)),
                "c_a_dev_max": float(np.max(score_dev)),
            }
        )
    return rows, diagnostics


def _summarize(metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    summary = (
        metrics.groupby(["model", "mode", "dataset", "policy"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            override_mean=("override_rate", "mean"),
            delta_accuracy_mean=("delta_accuracy_vs_setacsa", "mean"),
            delta_accuracy_std=("delta_accuracy_vs_setacsa", "std"),
            delta_override_mean=("delta_override_vs_setacsa", "mean"),
            n=("accuracy", "count"),
        )
        .sort_values(["model", "mode", "dataset", "delta_accuracy_mean"], ascending=[True, True, True, False])
    )
    pooled = (
        metrics.groupby(["model", "mode", "policy"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            override_mean=("override_rate", "mean"),
            delta_accuracy_mean=("delta_accuracy_vs_setacsa", "mean"),
            delta_accuracy_std=("delta_accuracy_vs_setacsa", "std"),
            delta_override_mean=("delta_override_vs_setacsa", "mean"),
            n=("accuracy", "count"),
        )
        .sort_values(["model", "mode", "delta_accuracy_mean"], ascending=[True, True, False])
    )
    summary.to_csv(out_dir / "prospective_confidence_gate_summary.csv", index=False)
    pooled.to_csv(out_dir / "prospective_confidence_gate_pooled.csv", index=False)
    return pooled


def _draw(pooled: pd.DataFrame, out_dir: Path) -> None:
    focus = pooled[
        pooled["policy"].isin(
            [
                "SetACSA",
                "CSAAB",
                "ProspectiveConfGateSetBudget",
                "ProspectiveConfGateDevBudgetCap",
                "ProspectiveConfGatePareto",
            ]
        )
    ].copy()
    for (model, mode), block in focus.groupby(["model", "mode"]):
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for _, r in block.iterrows():
            ax.scatter(float(r["delta_override_mean"]), float(r["delta_accuracy_mean"]), s=70)
            ax.text(
                float(r["delta_override_mean"]) + 0.002,
                float(r["delta_accuracy_mean"]) + 0.001,
                str(r["policy"]),
                fontsize=7,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.grid(True, linewidth=0.3, alpha=0.35)
        ax.set_xlabel("Delta override vs SetACSA")
        ax.set_ylabel("Delta held-out accuracy vs SetACSA")
        ax.set_title(f"Prospective confidence gate: {model} {mode}")
        fig.tight_layout()
        fig.savefig(out_dir / f"F_prospective_conf_gate_{model}_{mode}.png", dpi=1000, bbox_inches="tight")
        fig.savefig(out_dir / f"F_prospective_conf_gate_{model}_{mode}.pdf", bbox_inches="tight")
        plt.close(fig)


def _write_latex(pooled: pd.DataFrame, out_dir: Path) -> None:
    focus = pooled[
        pooled["policy"].isin(
            [
                "CSAAB",
                "ProspectiveConfGateSetBudget",
                "ProspectiveConfGateDevBudgetCap",
                "ProspectiveConfGatePareto",
            ]
        )
    ].copy()
    rows = []
    for _, r in focus.sort_values(["model", "mode", "policy"]).iterrows():
        model = str(r["model"]).replace("_", r"\_")
        policy = POLICY_LABELS.get(str(r["policy"]), str(r["policy"])).replace("_", r"\_")
        rows.append(
            f"{model} & {r['mode']} & {policy} & {float(r['delta_accuracy_mean']):+.4f} & "
            f"{float(r['delta_override_mean']):+.4f} & {float(r['accuracy_mean']):.4f} & "
            f"{float(r['override_mean']):.4f} \\\\"
        )
    tex = (
        r"""\begin{table*}[!t]
\centering
\caption{Prospective confidence-gate comparator on frozen posterior traces. Confidence thresholds were selected on development subjects only and evaluated on held-out subjects. Deltas are relative to SetACSA within model and mode.}
\label{tab:prospective-confidence-gate}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllrrrr}
\toprule
Model & Mode & Policy & $\Delta$ accuracy & $\Delta$ override & Accuracy & Override \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    (out_dir / "T_prospective_confidence_gate_table.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prospective validation-selected confidence-gate audit.")
    ap.add_argument("--results_root", type=Path, default=Path("results"))
    ap.add_argument("--base_exps", default="command_departure_suite,command_departure_transformer")
    ap.add_argument("--models", default="tcn_deep,transformer_small")
    ap.add_argument("--out_dir", type=Path, default=Path("results/prospective_confidence_gate_audit"))
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base_exps = [s.strip() for s in args.base_exps.split(",") if s.strip()]
    models = {s.strip() for s in args.models.split(",") if s.strip()} or None
    experiments = _discover(args.results_root, base_exps, models)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for exp in experiments:
        datasets = [p.name for p in sorted(exp.path.iterdir()) if p.is_dir() and (p / "sim.parquet").exists()]
        for ds in datasets:
            try:
                rr, dd = _process(exp, ds, n_splits=int(args.n_splits))
                rows.extend(rr)
                diagnostics.extend(dd)
            except Exception as exc:
                warnings.append({"exp_id": exp.exp_id, "dataset": ds, "warning": repr(exc)})
    if not rows:
        raise SystemExit("no prospective confidence-gate rows produced")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "prospective_confidence_gate_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out_dir / "prospective_confidence_gate_diagnostics.csv", index=False)
    if warnings:
        pd.DataFrame(warnings).to_csv(out_dir / "prospective_confidence_gate_warnings.csv", index=False)
    pooled = _summarize(metrics, out_dir)
    _draw(pooled, out_dir)
    _write_latex(pooled, out_dir)
    manifest = {
        "timestamp_utc": _utc_now(),
        "results_root": str(args.results_root),
        "base_exps": base_exps,
        "models": sorted(models) if models else [],
        "out_dir": str(out_dir),
        "n_experiments": int(len(experiments)),
        "n_rows": int(len(metrics)),
        "warnings": warnings,
    }
    (out_dir / "prospective_confidence_gate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
