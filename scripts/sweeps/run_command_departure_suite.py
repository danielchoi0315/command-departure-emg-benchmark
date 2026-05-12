#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class SuiteJob:
    mode: str
    model: str
    seed: int

    @property
    def tag(self) -> str:
        return f"{self.mode}__{self.model}__s{self.seed}"


def _run_cmd(cmd: list[str], *, env: dict[str, str], cwd: Path, log_path: Path, commands_log: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    commands_log.parent.mkdir(parents=True, exist_ok=True)
    cmd_s = " ".join(shlex.quote(c) for c in cmd)
    with commands_log.open("a", encoding="utf-8") as f:
        f.write(f"[{_utc_now()}] CMD: {cmd_s}\n")
        f.write(f"[{_utc_now()}] ENV_DIFF: {json.dumps({k: env[k] for k in sorted(env) if k.startswith('PU_') or k in {'DATA_ROOT','PAPER_ROOT','RUN_DEEP_IN_CORRECTNESS'}}, sort_keys=True)}\n")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] CMD: {cmd_s}\n")
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=log, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd_s}")


def _write_env_report(repo_root: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [os.environ.get("PYTHON_BIN", "python"), "scripts/utils/env_report.py", "--fs-target", str(repo_root), "--repo-hint", str(repo_root)]
    with out_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(repo_root), stdout=f, stderr=subprocess.STDOUT, check=True)


def _jobs_from_cfg(cfg: dict[str, Any]) -> list[SuiteJob]:
    suite = cfg["suite"]
    jobs: list[SuiteJob] = []
    for mode in suite["modes"]:
        for model in suite["pu_pred_model"]:
            for seed in suite["seeds"]:
                jobs.append(SuiteJob(mode=str(mode), model=str(model), seed=int(seed)))
    return jobs


def _exp_id(base_exp: str, job: SuiteJob) -> str:
    return f"{base_exp}_{job.mode}_{job.model}_s{job.seed}"


def _paper_root(base: Path, job: SuiteJob) -> Path:
    return base / "runs" / job.tag


def _marker_path(stage_state: Path, job: SuiteJob) -> Path:
    return stage_state / f"{job.tag}.done"


def _prepare_job_env(base_env: dict[str, str], suite_cfg: dict[str, Any], job: SuiteJob, paper_root: Path, data_root: Path) -> dict[str, str]:
    env = dict(base_env)
    env["DATA_ROOT"] = str(data_root)
    env["PAPER_ROOT"] = str(paper_root)
    env["PU_PRED_MODEL"] = job.model
    env["PU_MODELS"] = job.model

    mode_env = suite_cfg.get(f"{job.mode}_env", {}) or {}
    for k, v in mode_env.items():
        env[str(k)] = str(v)

    if job.model == "tcn_deep" and job.mode == "correctness":
        env["RUN_DEEP_IN_CORRECTNESS"] = "1"
    else:
        env["RUN_DEEP_IN_CORRECTNESS"] = "0"
    skip_preprocess = bool(suite_cfg.get("skip_preprocess", True))
    env["COMMAND_DEPARTURE_SKIP_PREPROCESS"] = "1" if skip_preprocess else "0"
    return env


def _run_jobs(
    *,
    repo_root: Path,
    datasets_yaml: Path,
    exp_yaml: Path,
    cfg: dict[str, Any],
    data_root: Path,
) -> tuple[list[dict[str, Any]], Path, Path]:
    suite = cfg["suite"]
    base_exp = str(suite["base_exp"])
    suite_results_root = repo_root / "results" / base_exp
    suite_results_root.mkdir(parents=True, exist_ok=True)
    stage_state = suite_results_root / "stage_state"
    stage_state.mkdir(parents=True, exist_ok=True)
    commands_log = suite_results_root / "commands.log"
    logs_dir = suite_results_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    _write_env_report(repo_root, suite_results_root / "ENV.txt")

    paper_root_base = repo_root / "artifacts" / "command_departure_suite"
    paper_root_base.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, Any]] = []
    jobs = _jobs_from_cfg(cfg)
    for job in jobs:
        marker = _marker_path(stage_state, job)
        exp_id = _exp_id(base_exp, job)
        run_paper_root = _paper_root(paper_root_base, job)
        run_log = logs_dir / f"{job.tag}.log"
        if marker.exists():
            try:
                completed.append(json.loads(marker.read_text(encoding="utf-8")))
            except Exception:
                completed.append({"job": job.tag, "exp_id": exp_id, "status": "done_marker_present"})
            continue

        env = _prepare_job_env(os.environ.copy(), suite, job, run_paper_root, data_root)
        cmd = [
            "bash",
            "run_all.sh",
            "--tier",
            str(int(suite["tier"])),
            "--exp",
            exp_id,
            "--paper_mode",
            str(int(suite["paper_mode"])),
            "--exp_mode",
            job.mode,
            "--datasets_yaml",
            str(datasets_yaml),
            "--exp_yaml",
            str(exp_yaml),
        ]
        _run_cmd(cmd, env=env, cwd=repo_root, log_path=run_log, commands_log=commands_log)

        rec = {
            "job": job.tag,
            "mode": job.mode,
            "model": job.model,
            "seed": int(job.seed),
            "exp_id": exp_id,
            "paper_root": str(run_paper_root),
            "timestamp_utc": _utc_now(),
            "status": "ok",
        }
        marker.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        completed.append(rec)

    return completed, suite_results_root, paper_root_base


def _collect_effect_rows(repo_root: Path, records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        exp_id = rec.get("exp_id")
        if not exp_id:
            continue
        tables_root = repo_root / rec["paper_root"] / "tables"
        subj_path = tables_root / "subject_policy_metrics.csv"
        if not subj_path.exists():
            continue
        subj = pd.read_csv(subj_path)
        if subj.empty:
            continue
        piv = subj.pivot_table(index=["dataset", "subject_id"], columns="policy", values="accuracy", aggfunc="mean")
        if "SetACSA" not in piv.columns:
            continue
        for policy in ["CSAAB", "CSAAB_BUDGET"]:
            if policy not in piv.columns:
                continue
            d = (piv[policy] - piv["SetACSA"]).dropna().groupby(level=0).mean().reset_index(name="delta_accuracy_subject_mean")
            d["policy"] = policy
            d["mode"] = rec["mode"]
            d["model"] = rec["model"]
            d["seed"] = int(rec["seed"])
            d["exp_id"] = exp_id
            rows.extend(d.to_dict("records"))
    return pd.DataFrame(rows)


def _write_summary(repo_root: Path, suite_results_root: Path, effects: pd.DataFrame) -> None:
    out_csv = suite_results_root / "suite_effects.csv"
    if effects.empty:
        out_csv.write_text("", encoding="utf-8")
        (suite_results_root / "RESULTS_SUMMARY.md").write_text("# RESULTS SUMMARY\n\nNo completed runs.\n", encoding="utf-8")
        return
    effects.to_csv(out_csv, index=False)

    md_lines = ["# RESULTS SUMMARY", "", "## Per-dataset effects (Δaccuracy vs SetACSA)"]
    per_ds = (
        effects.groupby(["dataset", "policy"], as_index=False)["delta_accuracy_subject_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    md_lines.append("")
    md_lines.append("| dataset | policy | mean_delta | std | n |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for _, row in per_ds.iterrows():
        md_lines.append(
            f"| {row['dataset']} | {row['policy']} | {float(row['mean']):.6f} | "
            f"{float(row['std']) if np.isfinite(float(row['std'])) else 0.0:.6f} | {int(row['count'])} |"
        )

    pooled = effects.groupby("policy", as_index=False)["delta_accuracy_subject_mean"].mean()
    md_lines.append("")
    md_lines.append("## Pooled effects")
    md_lines.append("")
    md_lines.append("| policy | pooled_mean_delta |")
    md_lines.append("|---|---:|")
    for _, row in pooled.iterrows():
        md_lines.append(f"| {row['policy']} | {float(row['delta_accuracy_subject_mean']):.6f} |")
    (suite_results_root / "RESULTS_SUMMARY.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _run_suite_parity(
    repo_root: Path,
    suite_cfg: dict[str, Any],
    records: list[dict[str, Any]],
    suite_results_root: Path,
    *,
    fail_hard: bool = True,
) -> dict[str, Any]:
    tol = float(suite_cfg.get("parity_tolerance", 1e-3))
    exp_map = {(r["mode"], r["model"], int(r["seed"])): r["exp_id"] for r in records if "exp_id" in r}
    failures = []
    parity_rows: list[dict[str, Any]] = []
    for model in suite_cfg["pu_pred_model"]:
        for seed in suite_cfg["seeds"]:
            key_t = ("throughput", model, int(seed))
            key_c = ("correctness", model, int(seed))
            if key_t not in exp_map or key_c not in exp_map:
                continue
            cmd = [
                "python",
                "scripts/utils/parity_check_command_departure_e2e.py",
                "--exp_correctness",
                exp_map[key_c],
                "--exp_throughput",
                exp_map[key_t],
                "--tol_policy",
                str(tol),
                "--tol_meta",
                str(tol),
            ]
            proc = subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
            parity_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "correctness_exp": exp_map[key_c],
                    "throughput_exp": exp_map[key_t],
                    "returncode": int(proc.returncode),
                    "output": proc.stdout.strip(),
                }
            )
            if proc.returncode != 0:
                failures.append({"model": model, "seed": int(seed), "output": proc.stdout.strip()})
    summary_path = suite_results_root / "mode_parity_summary.json"
    summary_path.write_text(json.dumps(parity_rows, indent=2), encoding="utf-8")
    parity = {
        "status": "pass" if not failures else "fail",
        "failures": int(len(failures)),
        "cells_compared": int(len(parity_rows)),
        "tolerance": tol,
        "summary_path": str(summary_path),
    }
    if failures and fail_hard:
        raise SystemExit(f"Suite parity failed for {len(failures)} mode/model/seed cells. See mode_parity_summary.json")
    return parity


def _package_outputs(repo_root: Path, base_exp: str, paper_root_base: Path, records: list[dict[str, Any]]) -> dict[str, str]:
    exp_dirs = sorted({r.get("exp_id") for r in records if r.get("exp_id")})
    if not exp_dirs:
        raise SystemExit("No completed runs to package.")

    release_zip = repo_root / "release_packet_command_departure_suite.zip"
    freeze_tar = repo_root / "CommandDepartureBenchmark_FREEZE_command_departure_suite.tar.gz"
    release_sha256 = Path(str(release_zip) + ".sha256")
    freeze_sha256 = Path(str(freeze_tar) + ".sha256")
    for p in [release_zip, freeze_tar, release_sha256, freeze_sha256]:
        if p.exists():
            p.unlink()

    # Hashes for key roots.
    subprocess.run(["bash", "scripts/utils/make_sha256sums_parallel.sh", "--root", str(paper_root_base)], cwd=str(repo_root), check=True)
    subprocess.run(["bash", "scripts/utils/make_sha256sums_parallel.sh", "--root", str(repo_root / "results" / base_exp)], cwd=str(repo_root), check=True)

    # Freeze tar.
    tar_cmd = ["tar", "-czf", str(freeze_tar), "config", "docs", "scripts", "run_all.sh", str(paper_root_base.relative_to(repo_root))]
    for exp_id in exp_dirs:
        tar_cmd.append(str((repo_root / "results" / exp_id).relative_to(repo_root)))
    tar_cmd.append(str((repo_root / "results" / base_exp).relative_to(repo_root)))
    subprocess.run(tar_cmd, cwd=str(repo_root), check=True)
    subprocess.run(["sha256sum", str(freeze_tar)], cwd=str(repo_root), stdout=freeze_sha256.open("w", encoding="utf-8"), check=True)

    # Release zip.
    pack_root = repo_root / "release" / "command_departure_suite"
    if pack_root.exists():
        subprocess.run(["rm", "-rf", str(pack_root)], check=True)
    pack_root.mkdir(parents=True, exist_ok=True)
    (pack_root / "results").mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-a", str(paper_root_base), str(pack_root)], check=True)
    subprocess.run(["cp", "-a", str(repo_root / "results" / base_exp), str(pack_root / "results")], check=True)
    for exp_id in exp_dirs:
        subprocess.run(["cp", "-a", str(repo_root / "results" / exp_id), str(pack_root / "results")], check=True)
    subprocess.run(["cp", "-a", str(repo_root / "config"), str(pack_root)], check=True)
    subprocess.run(["cp", "-a", str(repo_root / "scripts"), str(pack_root)], check=True)
    subprocess.run(["cp", "-a", str(repo_root / "docs"), str(pack_root)], check=True)
    subprocess.run(["zip", "-rq", str(release_zip), "."], cwd=str(pack_root), check=True)
    subprocess.run(["sha256sum", str(release_zip)], cwd=str(repo_root), stdout=release_sha256.open("w", encoding="utf-8"), check=True)

    return {
        "release_zip": str(release_zip),
        "release_sha256": str(release_sha256),
        "freeze_tar": str(freeze_tar),
        "freeze_sha256": str(freeze_sha256),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the command-departure benchmark suite (multi-seed/model/mode).")
    ap.add_argument("--suite_yaml", type=Path, default=Path("config/command_departure_suite.yaml"))
    ap.add_argument("--datasets_yaml", type=Path, default=Path("config/datasets.yaml"))
    ap.add_argument("--exp_yaml", type=Path, default=Path("config/experiment.yaml"))
    ap.add_argument("--data_root", type=Path, default=Path(os.environ.get("DATA_ROOT", "lake")))
    ap.add_argument("--skip_package", action="store_true")
    ap.add_argument(
        "--parity_soft_fail",
        action="store_true",
        help="Do not stop the suite on parity failure; still package outputs and record FAIL in the manifest.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg = _load_yaml(args.suite_yaml)
    records, suite_results_root, paper_root_base = _run_jobs(
        repo_root=repo_root,
        datasets_yaml=args.datasets_yaml,
        exp_yaml=args.exp_yaml,
        cfg=cfg,
        data_root=args.data_root,
    )
    effects = _collect_effect_rows(repo_root, records)
    _write_summary(repo_root, suite_results_root, effects)
    parity = _run_suite_parity(
        repo_root,
        cfg["suite"],
        records,
        suite_results_root,
        fail_hard=not args.parity_soft_fail,
    )

    package_paths: dict[str, str] = {}
    if not args.skip_package:
        package_paths = _package_outputs(repo_root, str(cfg["suite"]["base_exp"]), paper_root_base, records)

    out = {
        "suite_results_root": str(suite_results_root),
        "paper_root_base": str(paper_root_base),
        "records": len(records),
        "parity": parity,
        "package": package_paths,
    }
    (suite_results_root / "suite_manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
