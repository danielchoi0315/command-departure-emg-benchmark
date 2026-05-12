#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


POLICY_COLS = {
    "UserOnly": "g_hat_useronly",
    "AutoOnly": "g_hat_autoonly",
    "SetACSA": "g_hat_setacsa",
    "CSAAB": "g_hat_csaab",
    "CSAAB_BUDGET": "g_hat_csaab_budget",
    "RandomBudget": "g_hat_random_budget",
}

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_prob_matrix(series: pd.Series) -> np.ndarray:
    rows: list[np.ndarray] = []
    max_dim = 0
    for raw in series:
        arr = np.asarray(raw, dtype=float).reshape(-1)
        rows.append(arr)
        max_dim = max(max_dim, int(arr.size))
    out = np.full((len(rows), max_dim), 1e-12, dtype=float)
    for i, arr in enumerate(rows):
        if arr.size:
            out[i, : arr.size] = arr
    out = np.clip(out, 1e-12, None)
    return out / out.sum(axis=1, keepdims=True)


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, None)
    return -np.sum(p * np.log(p), axis=1)


def _load_pred_vectors(exp_root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pu_path = exp_root / dataset / "PRED" / "pu_pred.parquet"
    pa_path = exp_root / dataset / "PRED" / "pa_pred.parquet"
    if not pu_path.exists():
        raise FileNotFoundError(f"missing p_u predictions: {pu_path}")
    if not pa_path.exists():
        raise FileNotFoundError(f"missing p_a predictions: {pa_path}")
    return pd.read_parquet(pu_path), pd.read_parquet(pa_path)


def _merge_prediction_features(sim: pd.DataFrame, pu: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ["subject_id", "session_id", "trial_id", "window_id"] if c in sim.columns and c in pu.columns and c in pa.columns]
    if not keys:
        raise ValueError("no shared id columns for sim/prediction merge")
    pu_cols = keys + ["p_u_pred"]
    pa_cols = keys + ["p_a_pred"]
    out = sim.merge(pu[pu_cols], on=keys, how="left", validate="one_to_one")
    out = out.merge(pa[pa_cols], on=keys, how="left", validate="one_to_one")
    if len(out) != len(sim):
        raise ValueError("sim/prediction merge changed row count")
    if out["p_u_pred"].isna().any() or out["p_a_pred"].isna().any():
        raise ValueError("sim/prediction merge produced missing probability vectors")
    return out


def _features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    p_u = _as_prob_matrix(df["p_u_pred"])
    p_a = _as_prob_matrix(df["p_a_pred"])
    c_u = p_u.max(axis=1)
    c_a = p_a.max(axis=1)
    h_u = _entropy(p_u)
    h_a = _entropy(p_a)
    g_u = df["g_u"].to_numpy(dtype=int)
    g_a = df["g_a"].to_numpy(dtype=int)
    agree = (g_u == g_a).astype(float)
    workload = df["workload"].to_numpy(dtype=float) if "workload" in df.columns else np.full(len(df), 0.5)
    names = ["c_u", "c_a", "h_u", "h_a", "c_a_minus_c_u", "h_u_minus_h_a", "agree", "workload"]
    x = np.column_stack([c_u, c_a, h_u, h_a, c_a - c_u, h_u - h_a, agree, workload])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(float), names


def _subject_folds(subjects: np.ndarray, *, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.array(sorted(pd.Series(subjects).astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    chunks = np.array_split(unique, min(max(2, n_splits), len(unique)))
    folds = []
    for chunk in chunks:
        test_mask = np.isin(subjects.astype(str), chunk.astype(str))
        train_mask = ~test_mask
        if train_mask.any() and test_mask.any():
            folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return folds


def _accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y, dtype=int) == np.asarray(pred, dtype=int))) if len(y) else float("nan")


def _override(pred: np.ndarray, g_u: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred, dtype=int) != np.asarray(g_u, dtype=int))) if len(pred) else float("nan")


def _select_by_score(score: np.ndarray, g_u: np.ndarray, g_a: np.ndarray, *, target_override: float) -> np.ndarray:
    pred = np.asarray(g_u, dtype=int).copy()
    eligible = np.where(np.asarray(g_u, dtype=int) != np.asarray(g_a, dtype=int))[0]
    if eligible.size == 0 or target_override <= 0:
        return pred
    k = int(round(float(np.clip(target_override, 0.0, 1.0)) * len(pred)))
    k = max(0, min(k, int(eligible.size)))
    if k == 0:
        return pred
    chosen_local = np.argsort(-np.asarray(score, dtype=float)[eligible], kind="mergesort")[:k]
    chosen = eligible[chosen_local]
    pred[chosen] = np.asarray(g_a, dtype=int)[chosen]
    return pred


def _threshold_for_dev_override(score: np.ndarray, g_u: np.ndarray, g_a: np.ndarray, *, target_override: float) -> float:
    eligible = np.where(np.asarray(g_u, dtype=int) != np.asarray(g_a, dtype=int))[0]
    if eligible.size == 0 or target_override <= 0:
        return float("inf")
    k = int(round(float(np.clip(target_override, 0.0, 1.0)) * len(score)))
    k = max(1, min(k, int(eligible.size)))
    ranked = np.sort(np.asarray(score, dtype=float)[eligible])[::-1]
    return float(ranked[k - 1])


def _apply_threshold(score: np.ndarray, g_u: np.ndarray, g_a: np.ndarray, threshold: float) -> np.ndarray:
    pred = np.asarray(g_u, dtype=int).copy()
    mask = (np.asarray(g_u, dtype=int) != np.asarray(g_a, dtype=int)) & (np.asarray(score, dtype=float) >= float(threshold))
    pred[mask] = np.asarray(g_a, dtype=int)[mask]
    return pred


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (train_x - mu) / sd, (test_x - mu) / sd


def _learned_benefit_score(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y_bin = (train_y > 0).astype(int)
    if np.unique(y_bin).size < 2:
        constant = float(np.mean(y_bin))
        return np.full(len(train_x), constant), np.full(len(test_x), constant), {"model": "constant", "auc_dev": float("nan")}
    xtr, xte = _standardize(train_x, test_x)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    clf.fit(xtr, y_bin)
    train_score = clf.predict_proba(xtr)[:, 1]
    test_score = clf.predict_proba(xte)[:, 1]
    try:
        auc = float(roc_auc_score(y_bin, train_score))
    except Exception:
        auc = float("nan")
    return train_score, test_score, {"model": "logistic", "auc_dev": auc, "coef": clf.coef_[0].astype(float).tolist()}


def _choose_best_target(
    *,
    y_dev: np.ndarray,
    g_u_dev: np.ndarray,
    g_a_dev: np.ndarray,
    score_dev: np.ndarray,
    target_grid: list[float],
    max_override: float | None = None,
) -> tuple[float, dict[str, float]]:
    best: tuple[float, float, float] | None = None
    best_metrics: dict[str, float] = {}
    for target in target_grid:
        pred = _select_by_score(score_dev, g_u_dev, g_a_dev, target_override=target)
        acc = _accuracy(y_dev, pred)
        ov = _override(pred, g_u_dev)
        if max_override is not None and ov > max_override + 1e-12:
            continue
        # Primary: dev accuracy. Tie-break: lower override.
        key = (acc, -ov, -abs(target))
        if best is None or key > best:
            best = key
            best_metrics = {"target_override": float(target), "dev_accuracy": acc, "dev_override": ov}
    if not best_metrics:
        return 0.0, {"target_override": 0.0, "dev_accuracy": _accuracy(y_dev, g_u_dev), "dev_override": 0.0}
    return float(best_metrics["target_override"]), best_metrics


@dataclass(frozen=True)
class ExpInfo:
    exp_id: str
    mode: str
    model: str
    seed: int
    path: Path


def _discover_experiments(results_root: Path, *, base_exp: str, models: set[str] | None, modes: set[str] | None) -> list[ExpInfo]:
    exp_re = re.compile(rf"^{re.escape(base_exp)}_(?P<mode>correctness|throughput)_(?P<model>.+)_s(?P<seed>\d+)$")
    out: list[ExpInfo] = []
    for p in sorted(results_root.glob(f"{base_exp}_*")):
        if not p.is_dir():
            continue
        m = exp_re.match(p.name)
        if not m:
            continue
        mode = m.group("mode")
        model = m.group("model")
        seed = int(m.group("seed"))
        if models and model not in models:
            continue
        if modes and mode not in modes:
            continue
        out.append(ExpInfo(exp_id=p.name, mode=mode, model=model, seed=seed, path=p))
    return out


def _metrics_row(
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
    row = {
        "exp_id": exp.exp_id,
        "mode": exp.mode,
        "model": exp.model,
        "seed": int(exp.seed),
        "dataset": dataset,
        "split_id": int(split_id),
        "policy": policy,
        "accuracy": acc,
        "override_rate": ov,
        "delta_accuracy_vs_setacsa": acc - set_acc,
        "delta_override_vs_setacsa": ov - set_override,
        "n_windows": int(len(y)),
        "n_subjects": int(len(np.unique(g_u))) if len(g_u) else 0,
    }
    if extra:
        row.update(extra)
    return row


def _process_dataset(exp: ExpInfo, dataset: str, *, n_splits: int, target_grid: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sim_path = exp.path / dataset / "sim.parquet"
    if not sim_path.exists():
        return [], [{"exp_id": exp.exp_id, "dataset": dataset, "warning": f"missing {sim_path}"}]
    sim = pd.read_parquet(sim_path)
    pu, pa = _load_pred_vectors(exp.path, dataset)
    df = _merge_prediction_features(sim, pu, pa)
    x, feature_names = _features(df)

    y = df["g_star"].to_numpy(dtype=int)
    g_u = df["g_u"].to_numpy(dtype=int)
    g_a = df["g_a"].to_numpy(dtype=int)
    subjects = df["subject_id"].astype(str).to_numpy()
    benefit = (g_a == y).astype(int) - (g_u == y).astype(int)

    c_u = df["c_u"].to_numpy(dtype=float) if "c_u" in df.columns else x[:, 0]
    h_u = df["H_pu"].to_numpy(dtype=float) if "H_pu" in df.columns else x[:, 2]
    p_a = _as_prob_matrix(df["p_a_pred"])
    c_a = p_a.max(axis=1)
    score_map = {
        "TunedLowConfidenceBudget": -c_u,
        "TunedEntropyBudget": h_u,
        "TunedAssistConfidenceBudget": c_a - c_u,
    }

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    folds = _subject_folds(subjects, n_splits=n_splits, seed=exp.seed + sum(ord(ch) for ch in dataset))
    for split_id, (dev_idx, test_idx) in enumerate(folds):
        y_dev, y_test = y[dev_idx], y[test_idx]
        gu_dev, gu_test = g_u[dev_idx], g_u[test_idx]
        ga_dev, ga_test = g_a[dev_idx], g_a[test_idx]
        set_dev = df["g_hat_setacsa"].to_numpy(dtype=int)[dev_idx]
        set_test = df["g_hat_setacsa"].to_numpy(dtype=int)[test_idx]
        set_acc_dev = _accuracy(y_dev, set_dev)
        set_acc_test = _accuracy(y_test, set_test)
        set_ov_dev = _override(set_dev, gu_dev)
        set_ov_test = _override(set_test, gu_test)

        for policy, col in POLICY_COLS.items():
            if col in df.columns:
                pred = df[col].to_numpy(dtype=int)[test_idx]
                rows.append(
                    _metrics_row(
                        exp=exp,
                        dataset=dataset,
                        split_id=split_id,
                        policy=policy,
                        y=y_test,
                        g_u=gu_test,
                        pred=pred,
                        set_acc=set_acc_test,
                        set_override=set_ov_test,
                    )
                )

        for name, score_all in score_map.items():
            score_dev = score_all[dev_idx]
            score_test = score_all[test_idx]
            threshold = _threshold_for_dev_override(score_dev, gu_dev, ga_dev, target_override=set_ov_dev)
            pred = _apply_threshold(score_test, gu_test, ga_test, threshold)
            rows.append(
                _metrics_row(
                    exp=exp,
                    dataset=dataset,
                    split_id=split_id,
                    policy=name,
                    y=y_test,
                    g_u=gu_test,
                    pred=pred,
                    set_acc=set_acc_test,
                    set_override=set_ov_test,
                    extra={"dev_target_override": set_ov_dev, "selected_target_override": set_ov_dev, "dev_set_accuracy": set_acc_dev},
                )
            )

        score_dev, score_test, diag = _learned_benefit_score(x[dev_idx], benefit[dev_idx], x[test_idx])
        diagnostics.append(
            {
                "exp_id": exp.exp_id,
                "mode": exp.mode,
                "model": exp.model,
                "seed": int(exp.seed),
                "dataset": dataset,
                "split_id": int(split_id),
                "feature_names": feature_names,
                **diag,
            }
        )

        threshold = _threshold_for_dev_override(score_dev, gu_dev, ga_dev, target_override=set_ov_dev)
        pred = _apply_threshold(score_test, gu_test, ga_test, threshold)
        rows.append(
            _metrics_row(
                exp=exp,
                dataset=dataset,
                split_id=split_id,
                policy="TunedBenefitBudget",
                y=y_test,
                g_u=gu_test,
                pred=pred,
                set_acc=set_acc_test,
                set_override=set_ov_test,
                extra={"dev_target_override": set_ov_dev, "selected_target_override": set_ov_dev, "dev_set_accuracy": set_acc_dev},
            )
        )

        target, chosen = _choose_best_target(
            y_dev=y_dev,
            g_u_dev=gu_dev,
            g_a_dev=ga_dev,
            score_dev=score_dev,
            target_grid=target_grid,
            max_override=None,
        )
        pred = _select_by_score(score_test, gu_test, ga_test, target_override=target)
        rows.append(
            _metrics_row(
                exp=exp,
                dataset=dataset,
                split_id=split_id,
                policy="TunedBenefitPareto",
                y=y_test,
                g_u=gu_test,
                pred=pred,
                set_acc=set_acc_test,
                set_override=set_ov_test,
                extra=chosen,
            )
        )

        target, chosen = _choose_best_target(
            y_dev=y_dev,
            g_u_dev=gu_dev,
            g_a_dev=ga_dev,
            score_dev=score_dev,
            target_grid=target_grid,
            max_override=set_ov_dev,
        )
        pred = _select_by_score(score_test, gu_test, ga_test, target_override=target)
        rows.append(
            _metrics_row(
                exp=exp,
                dataset=dataset,
                split_id=split_id,
                policy="TunedBenefitDevBudgetCap",
                y=y_test,
                g_u=gu_test,
                pred=pred,
                set_acc=set_acc_test,
                set_override=set_ov_test,
                extra=chosen,
            )
        )

    return rows, diagnostics


def _write_summary(metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    group_cols = ["model", "mode", "dataset", "policy"]
    summary = (
        metrics.groupby(group_cols, as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            override_mean=("override_rate", "mean"),
            override_std=("override_rate", "std"),
            delta_accuracy_mean=("delta_accuracy_vs_setacsa", "mean"),
            delta_accuracy_std=("delta_accuracy_vs_setacsa", "std"),
            delta_override_mean=("delta_override_vs_setacsa", "mean"),
            n=("delta_accuracy_vs_setacsa", "count"),
        )
        .sort_values(["model", "mode", "dataset", "delta_accuracy_mean"], ascending=[True, True, True, False])
    )
    summary.to_csv(out_dir / "policy_tuning_summary.csv", index=False)

    pooled = (
        metrics.groupby(["model", "mode", "policy"], as_index=False)
        .agg(
            delta_accuracy_mean=("delta_accuracy_vs_setacsa", "mean"),
            delta_accuracy_std=("delta_accuracy_vs_setacsa", "std"),
            delta_override_mean=("delta_override_vs_setacsa", "mean"),
            accuracy_mean=("accuracy", "mean"),
            override_mean=("override_rate", "mean"),
            n=("delta_accuracy_vs_setacsa", "count"),
        )
        .sort_values(["model", "mode", "delta_accuracy_mean"], ascending=[True, True, False])
    )
    pooled.to_csv(out_dir / "policy_tuning_pooled.csv", index=False)
    return summary


def _draw_figures(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    focus = metrics[metrics["policy"].isin(["SetACSA", "CSAAB", "CSAAB_BUDGET", "TunedBenefitBudget", "TunedBenefitPareto", "TunedBenefitDevBudgetCap"])]
    for (model, mode), block in focus.groupby(["model", "mode"]):
        fig, ax = plt.subplots(figsize=(8, 5))
        agg = block.groupby("policy", as_index=False).agg(
            accuracy=("accuracy", "mean"),
            override=("override_rate", "mean"),
            dx=("delta_accuracy_vs_setacsa", "mean"),
        )
        for _, row in agg.iterrows():
            ax.scatter(row["override"], row["accuracy"], s=80)
            ax.text(row["override"], row["accuracy"], str(row["policy"]), fontsize=8, ha="left", va="bottom")
        ax.set_xlabel("Override rate / command-departure cost")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Validation-only policy frontier: {model} {mode}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / f"frontier_{model}_{mode}.png", dpi=220)
        plt.close(fig)


def _write_markdown(metrics: pd.DataFrame, out_dir: Path) -> None:
    pooled = pd.read_csv(out_dir / "policy_tuning_pooled.csv")
    lines = [
        "# Policy Tuning Summary",
        "",
        f"Generated: {_utc_now()}",
        "",
        "This analysis tunes arbitration policies on development subjects and evaluates on held-out subjects using frozen predictions only. No encoder, classifier, or assistant model is retrained.",
        "",
        "## Pooled Delta Accuracy vs SetACSA",
        "",
        "| model | mode | policy | mean_delta | mean_delta_override | accuracy | override | n |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in pooled.iterrows():
        lines.append(
            f"| {row['model']} | {row['mode']} | {row['policy']} | "
            f"{row['delta_accuracy_mean']:.6f} | {row['delta_override_mean']:.6f} | "
            f"{row['accuracy_mean']:.6f} | {row['override_mean']:.6f} | {int(row['n'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "A policy is paper-positive only if it improves held-out accuracy and does not materially increase override or command-departure cost relative to SetACSA. Unconstrained Pareto gains are useful but must be framed as higher-assistance operating points.",
        ]
    )
    (out_dir / "POLICY_TUNING_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validation-only Pareto tuning for command-departure benchmark arbitration policies.")
    ap.add_argument("--results_root", type=Path, default=Path("results"))
    ap.add_argument("--base_exp", default="command_departure_suite")
    ap.add_argument("--out_dir", type=Path, default=Path("results/policy_tuning"))
    ap.add_argument("--models", default="", help="Comma-separated model filter, e.g. tcn_deep or logreg,lda.")
    ap.add_argument("--modes", default="", help="Comma-separated mode filter, e.g. throughput,correctness.")
    ap.add_argument("--datasets", default="", help="Comma-separated dataset filter.")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--target_grid", default="0,0.02,0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.40,0.50")
    args = ap.parse_args()

    model_filter = {s.strip() for s in args.models.split(",") if s.strip()} or None
    mode_filter = {s.strip() for s in args.modes.split(",") if s.strip()} or None
    dataset_filter = {s.strip() for s in args.datasets.split(",") if s.strip()} or None
    target_grid = [float(s) for s in args.target_grid.split(",") if s.strip()]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    experiments = _discover_experiments(args.results_root, base_exp=args.base_exp, models=model_filter, modes=mode_filter)
    if not experiments:
        raise SystemExit("No matching experiment directories found.")

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for exp in experiments:
        datasets = [p.name for p in sorted(exp.path.iterdir()) if p.is_dir() and (p / "sim.parquet").exists()]
        if dataset_filter:
            datasets = [d for d in datasets if d in dataset_filter]
        for dataset in datasets:
            ds_rows, ds_diag = _process_dataset(exp, dataset, n_splits=args.n_splits, target_grid=target_grid)
            rows.extend(ds_rows)
            for item in ds_diag:
                if "warning" in item:
                    warnings.append(item)
                else:
                    diagnostics.append(item)

    if not rows:
        raise SystemExit("No tuning rows produced.")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "policy_tuning_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out_dir / "policy_tuning_diagnostics.csv", index=False)
    if warnings:
        pd.DataFrame(warnings).to_csv(out_dir / "policy_tuning_warnings.csv", index=False)
    summary = _write_summary(metrics, out_dir)
    _draw_figures(metrics, out_dir)
    _write_markdown(metrics, out_dir)
    manifest = {
        "timestamp_utc": _utc_now(),
        "results_root": str(args.results_root),
        "base_exp": args.base_exp,
        "out_dir": str(out_dir),
        "n_experiments": int(len(experiments)),
        "n_rows": int(len(metrics)),
        "n_summary_rows": int(len(summary)),
        "models": sorted(metrics["model"].unique().tolist()),
        "modes": sorted(metrics["mode"].unique().tolist()),
        "datasets": sorted(metrics["dataset"].unique().tolist()),
        "warnings": warnings,
    }
    (out_dir / "policy_tuning_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
