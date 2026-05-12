from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from command_departure_benchmark.eval.metrics import accuracy
from command_departure_benchmark.stats.ci import wilson_ci


POLICIES = {
    "UserOnly": "g_hat_useronly",
    "AutoOnly": "g_hat_autoonly",
    "ConfBlend": "g_hat_confblend",
    "SetACSA": "g_hat_setacsa",
    "RandomBudget": "g_hat_random_budget",
    "CSAAB": "g_hat_csaab",
    "CSAAB_BUDGET": "g_hat_csaab_budget",
    "CSAAB_Honly": "g_hat_csaab_entropy",
    "CSAAB_Wonly": "g_hat_csaab_workload",
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def ci95(values: np.ndarray) -> tuple[float, float, float]:
    vals = values[~np.isnan(values)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if len(vals) < 2:
        return mean, mean, mean
    se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


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
            g_u = sdf["g_u"].astype(int).to_numpy()
            n_windows = int(len(pred))
            acc_k = int(np.sum(pred == y))
            override_k = int(np.sum(pred != g_u))
            flip_n = int(max(0, len(pred) - 1))
            flip_k = int(np.sum(pred[1:] != pred[:-1])) if flip_n > 0 else 0
            high_unc_n = int(np.sum(high_unc))
            high_unc_k = int(np.sum(pred[high_unc] == y[high_unc])) if high_unc_n > 0 else 0
            high_w_n = int(np.sum(high_w))
            high_w_k = int(np.sum(pred[high_w] == y[high_w])) if high_w_n > 0 else 0
            rows.append(
                {
                    "dataset": dataset,
                    "subject_id": subject,
                    "policy": policy_name,
                    "accuracy": float(acc_k / n_windows) if n_windows > 0 else np.nan,
                    "accuracy_k": int(acc_k),
                    "accuracy_n": int(n_windows),
                    "macro_f1": macro_f1(y, pred),
                    "override_rate": float(override_k / n_windows) if n_windows > 0 else np.nan,
                    "override_k": int(override_k),
                    "override_n": int(n_windows),
                    "flip_rate": float(flip_k / flip_n) if flip_n > 0 else np.nan,
                    "flip_k": int(flip_k),
                    "flip_n": int(flip_n),
                    "acc_high_uncertainty": float(high_unc_k / high_unc_n) if high_unc_n > 0 else np.nan,
                    "acc_high_uncertainty_k": int(high_unc_k),
                    "acc_high_uncertainty_n": int(high_unc_n),
                    "acc_high_workload": float(high_w_k / high_w_n) if high_w_n > 0 else np.nan,
                    "acc_high_workload_k": int(high_w_k),
                    "acc_high_workload_n": int(high_w_n),
                }
            )
    return pd.DataFrame(rows)


def aggregate_subject_metrics(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prop_metrics = {
        "accuracy": ("accuracy_k", "accuracy_n"),
        "override_rate": ("override_k", "override_n"),
        "flip_rate": ("flip_k", "flip_n"),
        "acc_high_uncertainty": ("acc_high_uncertainty_k", "acc_high_uncertainty_n"),
        "acc_high_workload": ("acc_high_workload_k", "acc_high_workload_n"),
    }
    continuous_metrics = ["macro_f1"]
    for (dataset, policy), block in subject_df.groupby(["dataset", "policy"]):
        row = {"dataset": dataset, "policy": policy, "n_subjects": int(block["subject_id"].nunique())}
        for metric in continuous_metrics:
            mean, lo, hi = ci95(block[metric].to_numpy(dtype=float))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        for metric, (k_col, n_col) in prop_metrics.items():
            mean = float(block[metric].mean()) if metric in block.columns else float("nan")
            k_total = int(pd.to_numeric(block[k_col], errors="coerce").fillna(0).sum())
            n_total = int(pd.to_numeric(block[n_col], errors="coerce").fillna(0).sum())
            _, lo, hi = wilson_ci(k_total, n_total, alpha=0.05)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
            row[f"{metric}_k_total"] = k_total
            row[f"{metric}_n_total"] = n_total
        rows.append(row)
    return pd.DataFrame(rows)


def random_effects_meta(effects: np.ndarray, variances: np.ndarray) -> dict[str, float]:
    variances = np.clip(variances.astype(float), 1e-9, None)
    w = 1.0 / np.clip(variances, 1e-12, None)
    mu_fe = float(np.sum(w * effects) / np.sum(w))
    q = float(np.sum(w * (effects - mu_fe) ** 2))
    df = max(1, len(effects) - 1)
    c = float(np.sum(w) - (np.sum(w**2) / np.sum(w)))
    tau2 = max(0.0, (q - df) / max(c, 1e-12))
    w_re = 1.0 / (variances + tau2)
    mu_re = float(np.sum(w_re * effects) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    i2 = max(0.0, (q - df) / max(q, 1e-12)) * 100.0
    return {
        "effect_random": mu_re,
        "ci95_lo": mu_re - 1.96 * se_re,
        "ci95_hi": mu_re + 1.96 * se_re,
        "pred_interval_lo": mu_re - 1.96 * float(np.sqrt(tau2 + se_re**2)),
        "pred_interval_hi": mu_re + 1.96 * float(np.sqrt(tau2 + se_re**2)),
        "tau2": tau2,
        "i2": i2,
        "k_datasets": int(len(effects)),
    }


def meta_analysis(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for treatment in ["CSAAB", "CSAAB_BUDGET"]:
        for comparator in ["SetACSA", "ConfBlend"]:
            for endpoint in ["accuracy", "override_rate"]:
                eff = []
                var = []
                for dataset, ddf in subject_df.groupby("dataset"):
                    pivot = ddf.pivot(index="subject_id", columns="policy", values=endpoint)
                    if treatment not in pivot.columns or comparator not in pivot.columns:
                        continue
                    diff = (pivot[treatment] - pivot[comparator]).dropna().to_numpy(dtype=float)
                    if len(diff) < 2:
                        continue
                    eff.append(float(diff.mean()))
                    var.append(float(diff.var(ddof=1) / len(diff)))
                if len(eff) < 2:
                    continue
                out = random_effects_meta(np.array(eff), np.array(var))
                out.update(
                    {
                        "comparison": f"{treatment}_vs_{comparator}",
                        "endpoint": endpoint,
                    }
                )
                rows.append(out)
    return pd.DataFrame(rows)


def save_placeholder(fig_path: Path, title: str, note: str) -> None:
    fig = plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, note, ha="center", va="center")
    plt.title(title)
    plt.axis("off")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ensure_figure(fig_path: Path, source_path: Path, source_df: pd.DataFrame, draw_fn) -> None:
    source_df.to_excel(source_path, index=False)
    if source_df.empty:
        save_placeholder(fig_path, fig_path.stem, "No data available")
        return
    draw_fn(source_df, fig_path)


def draw_f1(df: pd.DataFrame, fig_path: Path) -> None:
    fig = plt.figure(figsize=(7, 5))
    for dataset, block in df.groupby("dataset"):
        plt.scatter(block["override_rate_mean"], block["accuracy_mean"], alpha=0.6, label=dataset)
    mean_block = df.groupby("policy", as_index=False)[["override_rate_mean", "accuracy_mean"]].mean()
    plt.scatter(mean_block["override_rate_mean"], mean_block["accuracy_mean"], c="black", marker="x", s=70, label="overall")
    for _, row in mean_block.iterrows():
        plt.text(row["override_rate_mean"], row["accuracy_mean"], row["policy"], fontsize=8)
    plt.xlabel("Override rate (lower better)")
    plt.ylabel("Accuracy (higher better)")
    plt.title("F1: Policy frontier")
    plt.legend(fontsize=7)
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_f2(df: pd.DataFrame, fig_path: Path) -> None:
    fig = plt.figure(figsize=(7, 5))
    for policy, block in df.groupby("policy"):
        block = block.sort_values("workload_bin")
        plt.plot(block["workload_bin"], block["accuracy"], marker="o", label=policy)
    plt.ylim(0, 1)
    plt.title("F2: Accuracy by workload bins")
    plt.xlabel("Workload bin")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=7)
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_f3(df: pd.DataFrame, fig_path: Path) -> None:
    fig = plt.figure(figsize=(7, 5))
    for policy, block in df.groupby("policy"):
        block = block.sort_values("uncertainty_bin")
        plt.plot(block["uncertainty_bin"], block["accuracy"], marker="o", label=policy)
    plt.ylim(0, 1)
    plt.title("F3: Accuracy by uncertainty bins")
    plt.xlabel("Uncertainty bin")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=7)
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_s1(df: pd.DataFrame, fig_path: Path) -> None:
    fig = plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    for (dataset, model), block in df.groupby(["dataset", "model"]):
        plt.plot(block["confidence"], block["accuracy"], marker="o", label=f"{dataset}:{model}")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("S1: Calibration reliability")
    plt.legend(fontsize=6)
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_s2(df: pd.DataFrame, fig_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    variants = sorted(df["policy_variant"].astype(str).unique().tolist())
    markers = {"CSAAB": "o", "CSAAB_BUDGET": "s"}
    for variant in variants:
        block = df[df["policy_variant"] == variant]
        plt.scatter(
            block["pa_accuracy_mean"],
            block["delta_accuracy_mean"],
            label=variant,
            alpha=0.85,
            marker=markers.get(variant, "o"),
            s=55,
        )
        for _, row in block.iterrows():
            plt.text(float(row["pa_accuracy_mean"]) + 0.002, float(row["delta_accuracy_mean"]) + 0.001, str(row["dataset"]), fontsize=7)

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("p_a held-out accuracy (mean across folds)")
    plt.ylabel("Δ accuracy vs SetACSA (subject mean)")
    plt.title("S2: Why it helps/hurts (p_a quality vs arbitration gain)")
    plt.legend(fontsize=8)
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_s3(df: pd.DataFrame, fig_path: Path) -> None:
    fig = plt.figure(figsize=(7, 5))
    for (dataset, policy), block in df.groupby(["dataset", "policy"]):
        block = block.sort_values("session_id")
        plt.plot(block["session_id"], block["accuracy"], marker="o", label=f"{dataset}:{policy}")
    plt.xlabel("Session/day")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("S3: Cross-day robustness")
    plt.legend(fontsize=7)
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--datasets_yaml", type=Path, required=True)
    ap.add_argument("--exp_yaml", type=Path, required=True)
    ap.add_argument("--paper_root", type=Path, default=Path(os.environ.get("PAPER_ROOT", "paper")))
    ap.add_argument("--results_root", type=Path, default=Path("results"))
    args = ap.parse_args()

    dsreg = load_yaml(args.datasets_yaml)["datasets"]
    exp = load_yaml(args.exp_yaml)["experiment"]
    exp_id = os.environ.get("EXP_ID", exp["id"])
    results_root = args.results_root / exp_id
    paper_root = args.paper_root
    figs_root = paper_root / "figures"
    src_root = paper_root / "source_data"
    tables_root = paper_root / "tables"
    figs_root.mkdir(parents=True, exist_ok=True)
    src_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    sim_data: dict[str, pd.DataFrame] = {}
    subject_rows = []
    for ds_id, info in dsreg.items():
        if int(info.get("tier", 99)) > args.tier:
            continue
        sim_path = results_root / ds_id / "sim.parquet"
        if not sim_path.exists():
            continue
        sim = pd.read_parquet(sim_path)
        sim_data[ds_id] = sim
        if "dataset" not in sim.columns:
            sim["dataset"] = ds_id
        if "subject_id" not in sim.columns:
            sim["subject_id"] = "S0"
        if "trial_id" not in sim.columns:
            sim["trial_id"] = "T0"
        if "window_id" not in sim.columns:
            sim["window_id"] = np.arange(len(sim))
        subject_rows.append(subject_policy_metrics(sim, dataset=ds_id))

    if subject_rows:
        subj_df = pd.concat(subject_rows, ignore_index=True)
        agg_df = aggregate_subject_metrics(subj_df)
        meta_df = meta_analysis(subj_df)
        f1_df = agg_df[agg_df["policy"].isin(["UserOnly", "ConfBlend", "SetACSA", "CSAAB", "CSAAB_BUDGET"])].copy()
    else:
        print("[ANALYZE] No sim outputs found. Writing placeholder paper artifacts.")
        subj_df = pd.DataFrame(columns=["dataset", "subject_id", "policy"])
        agg_df = pd.DataFrame(columns=["dataset", "policy"])
        meta_df = pd.DataFrame(columns=["comparison", "endpoint", "effect_random", "tau2", "i2"])
        f1_df = pd.DataFrame(columns=["dataset", "policy", "override_rate_mean", "accuracy_mean"])

    # Subject-level CI + pooled-window Wilson CI summaries.
    pooled_rows = []
    for ds_id, sim in sim_data.items():
        y = sim["g_star"].astype(int).to_numpy()
        g_u = sim["g_u"].astype(int).to_numpy()
        n_rows = int(len(sim))
        for policy_name, col in POLICIES.items():
            if col not in sim.columns:
                continue
            pred = sim[col].astype(int).to_numpy()
            acc_k = int(np.sum(pred == y))
            over_k = int(np.sum(pred != g_u))
            _, acc_lo, acc_hi = wilson_ci(acc_k, n_rows, alpha=0.05)
            _, over_lo, over_hi = wilson_ci(over_k, n_rows, alpha=0.05)
            pooled_rows.append(
                {
                    "dataset": ds_id,
                    "policy": policy_name,
                    "window_n": n_rows,
                    "accuracy_window_pooled": float(acc_k / n_rows) if n_rows > 0 else np.nan,
                    "accuracy_window_ci95_lo": acc_lo,
                    "accuracy_window_ci95_hi": acc_hi,
                    "override_window_pooled": float(over_k / n_rows) if n_rows > 0 else np.nan,
                    "override_window_ci95_lo": over_lo,
                    "override_window_ci95_hi": over_hi,
                }
            )
    pooled_df = pd.DataFrame(pooled_rows)
    subject_pooled = agg_df.merge(pooled_df, on=["dataset", "policy"], how="outer")

    subj_df.to_csv(tables_root / "subject_policy_metrics.csv", index=False)
    agg_df.to_csv(tables_root / "policy_summary.csv", index=False)
    agg_df.to_excel(src_root / "policy_summary.xlsx", index=False)
    subject_pooled.to_csv(tables_root / "policy_summary_subject_pooled.csv", index=False)
    subject_pooled.to_excel(src_root / "policy_summary_subject_pooled.xlsx", index=False)
    meta_df.to_csv(tables_root / "meta_analysis.csv", index=False)
    meta_df.to_excel(src_root / "meta_analysis.xlsx", index=False)

    # F2/F3 stratification.
    strat_rows_w = []
    strat_rows_h = []
    for ds_id, sim in sim_data.items():
        sim = sim.copy()
        if "workload" in sim.columns:
            sim["workload_bin"] = pd.qcut(sim["workload"].rank(method="first"), 3, labels=["low", "med", "high"])
        else:
            sim["workload_bin"] = "med"
        if "H_pu" in sim.columns:
            sim["uncertainty_bin"] = pd.qcut(sim["H_pu"].rank(method="first"), 3, labels=["low", "med", "high"])
        else:
            sim["uncertainty_bin"] = "med"
        y = sim["g_star"].astype(int).to_numpy()
        for policy_name, col in POLICIES.items():
            if col not in sim.columns:
                continue
            pred = sim[col].astype(int).to_numpy()
            for bin_name, bdf in sim.groupby("workload_bin", observed=False):
                idx = bdf.index.to_numpy()
                strat_rows_w.append(
                    {"dataset": ds_id, "policy": policy_name, "workload_bin": str(bin_name), "accuracy": accuracy(y[idx], pred[idx])}
                )
            for bin_name, bdf in sim.groupby("uncertainty_bin", observed=False):
                idx = bdf.index.to_numpy()
                strat_rows_h.append(
                    {
                        "dataset": ds_id,
                        "policy": policy_name,
                        "uncertainty_bin": str(bin_name),
                        "accuracy": accuracy(y[idx], pred[idx]),
                    }
                )
    f2_df = pd.DataFrame(strat_rows_w)
    f3_df = pd.DataFrame(strat_rows_h)

    # S1 calibration curves from p_u training artifacts.
    s1_rows = []
    for ds_id in sim_data.keys():
        cpath = results_root / ds_id / "calibration.json"
        if not cpath.exists():
            continue
        try:
            entries = json.loads(cpath.read_text())
        except Exception:
            continue
        for entry in entries:
            model = entry.get("model", "unknown")
            fold = entry.get("fold", -1)
            for point in entry.get("reliability_curve", []):
                s1_rows.append(
                    {
                        "dataset": ds_id,
                        "model": model,
                        "fold": fold,
                        "bin_lo": point.get("bin_lo"),
                        "bin_hi": point.get("bin_hi"),
                        "count": point.get("count"),
                        "accuracy": point.get("accuracy"),
                        "confidence": point.get("confidence"),
                    }
                )
    s1_df = pd.DataFrame(s1_rows)

    # S2: p_a quality vs Δaccuracy (CSAAB / CSAAB_BUDGET vs SetACSA).
    pa_quality_rows = []
    for ds_id in sim_data.keys():
        pa_fold_path = results_root / ds_id / "pa_fold_metrics.json"
        if not pa_fold_path.exists():
            continue
        try:
            fold_rows = json.loads(pa_fold_path.read_text())
        except Exception:
            continue
        if not isinstance(fold_rows, list):
            continue
        for row in fold_rows:
            if not isinstance(row, dict):
                continue
            pa_quality_rows.append(
                {
                    "dataset": ds_id,
                    "method": str(row.get("method", "unknown")),
                    "fold": int(row.get("fold", -1)),
                    "pa_accuracy": float(row.get("pa_accuracy", np.nan)),
                    "pa_nll": float(row.get("pa_nll", np.nan)),
                    "coverage": float(row.get("coverage", np.nan)),
                    "n_test_subjects": float(row.get("n_test_subjects", np.nan)),
                }
            )
    pa_quality_df = pd.DataFrame(pa_quality_rows)
    if not pa_quality_df.empty and not subj_df.empty:
        pa_dataset = (
            pa_quality_df.groupby("dataset", as_index=False)
            .agg(
                pa_accuracy_mean=("pa_accuracy", "mean"),
                pa_nll_mean=("pa_nll", "mean"),
                coverage_mean=("coverage", "mean"),
            )
            .sort_values("dataset", kind="stable")
            .reset_index(drop=True)
        )
        piv = subj_df.pivot_table(index=["dataset", "subject_id"], columns="policy", values="accuracy", aggfunc="mean")
        delta_rows = []
        for policy_variant in ["CSAAB", "CSAAB_BUDGET"]:
            if policy_variant not in piv.columns or "SetACSA" not in piv.columns:
                continue
            d = (piv[policy_variant] - piv["SetACSA"]).dropna().groupby(level=0).mean().reset_index(name="delta_accuracy_mean")
            d["policy_variant"] = policy_variant
            delta_rows.append(d)
        if delta_rows:
            delta_df = pd.concat(delta_rows, ignore_index=True)
            s2_df = delta_df.merge(pa_dataset, on="dataset", how="inner")
        else:
            s2_df = pd.DataFrame(
                columns=[
                    "dataset",
                    "policy_variant",
                    "delta_accuracy_mean",
                    "pa_accuracy_mean",
                    "pa_nll_mean",
                    "coverage_mean",
                ]
            )
    else:
        s2_df = pd.DataFrame(
            columns=[
                "dataset",
                "policy_variant",
                "delta_accuracy_mean",
                "pa_accuracy_mean",
                "pa_nll_mean",
                "coverage_mean",
            ]
        )

    # S3 session-wise robustness for GRABMyo/Hyser.
    s3_rows = []
    for ds_id in ["physionet_grabmyo", "physionet_hyser"]:
        sim = sim_data.get(ds_id)
        if sim is None or "session_id" not in sim.columns:
            continue
        y = sim["g_star"].astype(int).to_numpy()
        for policy_name in ["SetACSA", "CSAAB"]:
            col = POLICIES[policy_name]
            if col not in sim.columns:
                continue
            pred = sim[col].astype(int).to_numpy()
            for session_id, block in sim.groupby("session_id"):
                idx = block.index.to_numpy()
                s3_rows.append(
                    {
                        "dataset": ds_id,
                        "policy": policy_name,
                        "session_id": str(session_id),
                        "accuracy": accuracy(y[idx], pred[idx]),
                    }
                )
    s3_df = pd.DataFrame(s3_rows)

    # Render figures + write source data.
    ensure_figure(figs_root / "F1.png", src_root / "F1.xlsx", f1_df, draw_f1)
    ensure_figure(figs_root / "F2.png", src_root / "F2.xlsx", f2_df, draw_f2)
    ensure_figure(figs_root / "F3.png", src_root / "F3.xlsx", f3_df, draw_f3)
    ensure_figure(figs_root / "S1.png", src_root / "S1.xlsx", s1_df, draw_s1)
    ensure_figure(figs_root / "S2.png", src_root / "S2.xlsx", s2_df, draw_s2)
    ensure_figure(figs_root / "S3.png", src_root / "S3.xlsx", s3_df, draw_s3)

    # Backward-compatible figure alias for toy demo.
    try:
        (figs_root / "F_frontier.png").write_bytes((figs_root / "F1.png").read_bytes())
    except Exception:
        pass

    paper_prefix = paper_root.as_posix().rstrip("/")
    figure_manifest = [
        {
            "figure_id": fig_id,
            "figure_path": f"{paper_prefix}/figures/{fig_id}.png",
            "source_data_path": f"{paper_prefix}/source_data/{fig_id}.xlsx",
        }
        for fig_id in ["F1", "F2", "F3", "S1", "S2", "S3"]
    ]
    (paper_root / "figure_manifest.json").write_text(json.dumps(figure_manifest, indent=2))

    print("[ANALYZE] Wrote paper artifacts (figures, source data, tables, meta-analysis).")


if __name__ == "__main__":
    main()
