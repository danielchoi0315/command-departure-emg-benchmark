#!/usr/bin/env python3
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

from command_departure_benchmark.eval.metrics import accuracy


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


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _permute_within_subject(y: np.ndarray, subjects: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = y.copy()
    for sid in np.unique(subjects):
        idx = np.where(subjects == sid)[0]
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def _subject_policy_metrics(sim: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for subject, sdf in sim.groupby("subject_id"):
        sdf = sdf.sort_values(["trial_id", "window_id"], kind="stable")
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
                    "accuracy": accuracy(y, pred),
                    "override_rate": float(np.mean(pred != g_u)),
                    "flip_rate": float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _aggregate_subject_metrics(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, policy), block in subject_df.groupby(["dataset", "policy"]):
        rows.append(
            {
                "dataset": dataset,
                "policy": policy,
                "accuracy_mean": float(block["accuracy"].mean()),
                "override_rate_mean": float(block["override_rate"].mean()),
                "flip_rate_mean": float(block["flip_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _update_figure_manifest(paper_root: Path) -> None:
    manifest_path = paper_root / "figure_manifest.json"
    entries: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []
    by_id = {e.get("figure_id"): e for e in entries if isinstance(e, dict) and e.get("figure_id")}
    prefix = paper_root.as_posix().rstrip("/")
    by_id["S10"] = {
        "figure_id": "S10",
        "figure_path": f"{prefix}/figures/S10_e2e_audit.png",
        "source_data_path": f"{prefix}/source_data/S10_e2e_audit.xlsx",
    }
    ordered = [by_id[k] for k in sorted(by_id.keys(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 999))]
    manifest_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def _draw_s10(
    *,
    ds_df: pd.DataFrame,
    recompute_max_diff: float,
    recompute_tol: float,
    fig_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if ds_df.empty:
        for ax in axes.ravel():
            ax.text(0.5, 0.5, "No e2e datasets", ha="center", va="center")
            ax.axis("off")
        fig.savefig(fig_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    x = np.arange(len(ds_df))

    ax = axes[0, 0]
    ax.bar(x, ds_df["useronly_acc"].to_numpy(dtype=float))
    ax.axhline(float(ds_df["useronly_threshold"].iloc[0]), color="red", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_df["dataset"], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("UserOnly accuracy (must not saturate)")

    ax = axes[0, 1]
    ax.bar(x - 0.15, ds_df["perm_acc_useronly"].to_numpy(dtype=float), width=0.3, label="UserOnly")
    ax.bar(x + 0.15, ds_df["perm_acc_csaab"].to_numpy(dtype=float), width=0.3, label="CSAAB")
    ax.plot(x, ds_df["perm_threshold"].to_numpy(dtype=float), color="black", marker="o", linestyle="--", label="threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_df["dataset"], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Permutation collapse")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(x - 0.15, ds_df["engage_override_rate"].to_numpy(dtype=float), width=0.3, label="override_rate")
    ax.bar(x + 0.15, ds_df["engage_gate_rate"].to_numpy(dtype=float), width=0.3, label="gate_rate")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_df["dataset"], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Engagement nonzero")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar([0], [recompute_max_diff], width=0.4, label="max_abs_diff")
    ax.axhline(recompute_tol, color="red", linestyle="--", linewidth=1, label="tolerance")
    ax.set_xlim(-0.8, 0.8)
    ax.set_xticks([0])
    ax.set_xticklabels(["recompute"])
    ax.set_yscale("log")
    ax.set_title("Recompute consistency")
    ax.legend(fontsize=8)

    fig.suptitle("S10: Command-departure benchmark end-to-end integrity audit")
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Command-departure benchmark end-to-end audit checks (fail-closed).")
    ap.add_argument("--exp", required=True, help="Experiment id under results/")
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--datasets_yaml", type=Path, default=Path("config/datasets.yaml"))
    ap.add_argument("--paper_root", type=Path, default=Path(os.environ.get("PAPER_ROOT", "artifacts/command_departure")))
    ap.add_argument("--results_root", type=Path, default=Path("results"))
    ap.add_argument("--data_root", type=Path, default=Path(os.environ.get("DATA_ROOT", "lake")))
    ap.add_argument("--useronly_saturation_threshold", type=float, default=0.98)
    ap.add_argument("--perm_margin", type=float, default=0.08)
    ap.add_argument("--engagement_min_rate", type=float, default=1e-6)
    ap.add_argument("--recompute_tolerance", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    dsreg = _load_yaml(args.datasets_yaml)["datasets"]
    exp_root = args.results_root / args.exp
    if not exp_root.exists():
        raise SystemExit(f"missing experiment root: {exp_root}")

    tables_root = args.paper_root / "tables"
    src_root = args.paper_root / "source_data"
    figs_root = args.paper_root / "figures"
    tables_root.mkdir(parents=True, exist_ok=True)
    src_root.mkdir(parents=True, exist_ok=True)
    figs_root.mkdir(parents=True, exist_ok=True)

    ds_rows: list[dict[str, Any]] = []
    fail_reasons: list[str] = []
    subj_rows: list[pd.DataFrame] = []

    for ds_id, info in dsreg.items():
        if int(info.get("tier", 99)) > args.tier:
            continue
        role = str(info.get("arbitration_role", "full")).strip().lower() or "full"
        if role in {"p_a_only", "workload_only"}:
            continue
        windows_path = args.data_root / "derived" / ds_id / str(info.get("version", "unknown")) / "windows.parquet"
        sim_path = exp_root / ds_id / "sim.parquet"
        if not sim_path.exists():
            # Only require sim outputs for datasets that were actually materialized.
            if windows_path.exists():
                fail_reasons.append(f"{ds_id}: missing sim.parquet for e2e audit")
            continue
        sim = pd.read_parquet(sim_path)
        required_cols = {
            "g_star",
            "subject_id",
            "trial_id",
            "window_id",
            "g_u",
            "g_hat_useronly",
            "g_hat_csaab",
            "c_u",
            "tau_dyn",
        }
        missing = sorted(required_cols - set(sim.columns))
        if missing:
            fail_reasons.append(f"{ds_id}: missing required sim columns {missing}")
            continue

        y = sim["g_star"].astype(int).to_numpy()
        subjects = sim["subject_id"].astype(str).to_numpy()
        useronly_pred = sim["g_hat_useronly"].astype(int).to_numpy()
        csaab_pred = sim["g_hat_csaab"].astype(int).to_numpy()
        useronly_acc = accuracy(y, useronly_pred)
        sat_pass = bool(useronly_acc < float(args.useronly_saturation_threshold))
        if not sat_pass:
            fail_reasons.append(
                f"{ds_id}: useronly accuracy saturated ({useronly_acc:.6f} >= {args.useronly_saturation_threshold:.6f})"
            )

        ds_seed = int(args.seed + sum(ord(ch) for ch in ds_id))
        y_perm = _permute_within_subject(y, subjects, seed=ds_seed)
        perm_acc_user = accuracy(y_perm, useronly_pred)
        perm_acc_csaab = accuracy(y_perm, csaab_pred)
        n_classes = int(np.max(y) + 1) if len(y) else 1
        chance_uniform = 1.0 / max(1, n_classes)
        chance_majority = float(pd.Series(y).value_counts(normalize=True).max()) if len(y) else chance_uniform
        collapse_thr = float(max(chance_uniform, chance_majority) + args.perm_margin)
        perm_pass = bool(perm_acc_user <= collapse_thr and perm_acc_csaab <= collapse_thr)
        if not perm_pass:
            fail_reasons.append(
                f"{ds_id}: permutation not collapsed (user={perm_acc_user:.6f}, csaab={perm_acc_csaab:.6f}, thr={collapse_thr:.6f})"
            )

        gate_rate = float(np.mean(sim["c_u"].to_numpy(dtype=float) < sim["tau_dyn"].to_numpy(dtype=float)))
        override_rate = float(np.mean(sim["g_hat_csaab"].astype(int).to_numpy() != sim["g_u"].astype(int).to_numpy()))
        engage_pass = bool(gate_rate > args.engagement_min_rate and override_rate > args.engagement_min_rate)
        if not engage_pass:
            fail_reasons.append(
                f"{ds_id}: engagement too low (gate={gate_rate:.6f}, override={override_rate:.6f})"
            )

        subj_rows.append(_subject_policy_metrics(sim, dataset=ds_id))
        ds_rows.append(
            {
                "dataset": ds_id,
                "useronly_acc": float(useronly_acc),
                "useronly_threshold": float(args.useronly_saturation_threshold),
                "useronly_pass": sat_pass,
                "perm_acc_useronly": float(perm_acc_user),
                "perm_acc_csaab": float(perm_acc_csaab),
                "perm_threshold": collapse_thr,
                "perm_pass": perm_pass,
                "engage_gate_rate": gate_rate,
                "engage_override_rate": override_rate,
                "engagement_pass": engage_pass,
            }
        )

    ds_df = pd.DataFrame(ds_rows)

    recompute_pass = False
    recompute_max_diff = float("nan")
    if not subj_rows:
        fail_reasons.append("no valid datasets available for recompute consistency")
    else:
        recomputed = _aggregate_subject_metrics(pd.concat(subj_rows, ignore_index=True))
        summary_path = tables_root / "policy_summary.csv"
        if not summary_path.exists():
            fail_reasons.append(f"missing policy summary table: {summary_path}")
        else:
            table_df = pd.read_csv(summary_path)
            key = ["dataset", "policy"]
            cols = ["accuracy_mean", "override_rate_mean", "flip_rate_mean"]
            merged = recomputed.merge(
                table_df[key + cols],
                on=key,
                how="outer",
                suffixes=("_re", "_tb"),
                indicator=True,
            )
            if (merged["_merge"] != "both").any():
                fail_reasons.append("recompute consistency row mismatch between sim-derived and policy_summary")
            diffs = []
            for _, row in merged[merged["_merge"] == "both"].iterrows():
                for col in cols:
                    diffs.append(abs(float(row[f"{col}_re"]) - float(row[f"{col}_tb"])))
            recompute_max_diff = float(max(diffs)) if diffs else float("nan")
            recompute_pass = bool(np.isfinite(recompute_max_diff) and recompute_max_diff <= args.recompute_tolerance)
            if not recompute_pass:
                fail_reasons.append(
                    f"recompute consistency failed (max_diff={recompute_max_diff:.3e} tol={args.recompute_tolerance:.3e})"
                )

    audit_rows: list[dict[str, Any]] = []
    for _, row in ds_df.iterrows():
        audit_rows.append(
            {
                "section": "useronly_saturation",
                "dataset": row["dataset"],
                "value": row["useronly_acc"],
                "threshold": row["useronly_threshold"],
                "pass": bool(row["useronly_pass"]),
            }
        )
        audit_rows.append(
            {
                "section": "permutation_collapse",
                "dataset": row["dataset"],
                "value": max(float(row["perm_acc_useronly"]), float(row["perm_acc_csaab"])),
                "threshold": float(row["perm_threshold"]),
                "pass": bool(row["perm_pass"]),
            }
        )
        audit_rows.append(
            {
                "section": "engagement_nonzero",
                "dataset": row["dataset"],
                "value": min(float(row["engage_gate_rate"]), float(row["engage_override_rate"])),
                "threshold": float(args.engagement_min_rate),
                "pass": bool(row["engagement_pass"]),
            }
        )
    audit_rows.append(
        {
            "section": "recompute_consistency",
            "dataset": "__all__",
            "value": recompute_max_diff,
            "threshold": float(args.recompute_tolerance),
            "pass": bool(recompute_pass),
        }
    )
    audit_df = pd.DataFrame(audit_rows)

    table_path = tables_root / "command_departure_audit.csv"
    fig_path = figs_root / "S10_e2e_audit.png"
    src_path = src_root / "S10_e2e_audit.xlsx"
    audit_df.to_csv(table_path, index=False)
    with pd.ExcelWriter(src_path, engine="openpyxl") as w:
        pd.DataFrame([["Linked figure: S10"]]).to_excel(w, sheet_name="data", index=False, header=False)
        audit_df.to_excel(w, sheet_name="data", index=False, startrow=2)
    _draw_s10(ds_df=ds_df, recompute_max_diff=recompute_max_diff, recompute_tol=args.recompute_tolerance, fig_path=fig_path)
    _update_figure_manifest(args.paper_root)

    report = {
        "exp": args.exp,
        "table": str(table_path),
        "figure": str(fig_path),
        "source_data": str(src_path),
        "n_dataset_rows": int(len(ds_df)),
        "n_fail_reasons": int(len(fail_reasons)),
        "failed_reasons": fail_reasons,
        "recompute_max_diff": recompute_max_diff,
        "recompute_tolerance": float(args.recompute_tolerance),
    }
    report_path = exp_root / "command_departure_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if fail_reasons:
        print("[COMMAND_DEPARTURE_AUDIT] FAIL")
        for reason in fail_reasons:
            print(f"- {reason}")
        print(f"[COMMAND_DEPARTURE_AUDIT] report: {report_path}")
        raise SystemExit(1)

    print(f"[COMMAND_DEPARTURE_AUDIT] PASS (report: {report_path})")


if __name__ == "__main__":
    main()
