from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import os

from command_departure_benchmark.arbitration.ali import ali_from_tau
from command_departure_benchmark.arbitration.policies import CSAABParams, confblend, csaab_tau
from command_departure_benchmark.arbitration.tau_calibration import (
    DynamicGateRateParams,
    calibrate_budgeted_dynamic_rates,
    fit_tau_from_train_confidence,
    target_gate_rate_dyn,
)
from command_departure_benchmark.eval.splits import SubjectKFold
from command_departure_benchmark.schema import entropy


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    mat = np.clip(mat, 1e-12, None)
    return mat / mat.sum(axis=1, keepdims=True)


def _as_prob_matrix(series: pd.Series) -> np.ndarray:
    return _normalize_rows(np.vstack(series.apply(lambda v: np.asarray(v, dtype=float))))


def _quantile_linear(values: np.ndarray, q: np.ndarray | float) -> np.ndarray:
    try:
        return np.quantile(values, q=q, method="linear")
    except TypeError:
        # NumPy<1.22 compatibility.
        return np.quantile(values, q=q, interpolation="linear")


def _select_id_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["dataset", "subject_id", "session_id", "trial_id", "window_id", "g_star"]
    return [c for c in preferred if c in df.columns]


def _prediction_join_keys(windows: pd.DataFrame, pu: pd.DataFrame, pa: pd.DataFrame) -> list[str]:
    preferred = ["dataset", "subject_id", "session_id", "trial_id", "window_id"]
    keys = [c for c in preferred if c in windows.columns and c in pu.columns and c in pa.columns]
    if not keys:
        raise ValueError("No shared identifier columns for joining windows with pu_pred/pa_pred.")
    return keys


def _merge_predicted_priors(windows: pd.DataFrame, pu_path: Path, pa_path: Path) -> pd.DataFrame:
    if not pu_path.exists():
        raise FileNotFoundError(f"Missing required p_u predictions: {pu_path}")
    if not pa_path.exists():
        raise FileNotFoundError(f"Missing required p_a predictions: {pa_path}")

    pu = pd.read_parquet(pu_path)
    pa = pd.read_parquet(pa_path)
    if "p_u_pred" not in pu.columns:
        raise ValueError(f"{pu_path} missing p_u_pred column")
    if "p_a_pred" not in pa.columns:
        raise ValueError(f"{pa_path} missing p_a_pred column")

    keys = _prediction_join_keys(windows, pu, pa)
    pu_id = pu[keys + ["p_u_pred"]].copy()
    pa_id = pa[keys + ["p_a_pred"]].copy()
    if pu_id.duplicated(subset=keys).any():
        dup_n = int(pu_id.duplicated(subset=keys).sum())
        raise ValueError(f"{pu_path} has duplicate key rows (n={dup_n})")
    if pa_id.duplicated(subset=keys).any():
        dup_n = int(pa_id.duplicated(subset=keys).sum())
        raise ValueError(f"{pa_path} has duplicate key rows (n={dup_n})")

    merged = windows.merge(pu_id, on=keys, how="left", validate="one_to_one")
    merged = merged.merge(pa_id, on=keys, how="left", validate="one_to_one")
    if len(merged) != len(windows):
        raise ValueError("windows/prediction merge changed row count")
    if merged["p_u_pred"].isna().any():
        n_miss = int(merged["p_u_pred"].isna().sum())
        raise ValueError(f"Missing p_u_pred for {n_miss} windows after join")
    if merged["p_a_pred"].isna().any():
        n_miss = int(merged["p_a_pred"].isna().sum())
        raise ValueError(f"Missing p_a_pred for {n_miss} windows after join")

    merged["p_u"] = merged["p_u_pred"]
    merged["p_a"] = merged["p_a_pred"]
    return merged


def _deterministic_tie_jitter(df: pd.DataFrame, *, seed: int) -> np.ndarray:
    """Deterministic [0,1) jitter from stable row identifiers.

    This is used only to break exact confidence ties without using labels.
    """
    key_cols = [c for c in ["subject_id", "session_id", "trial_id", "window_id"] if c in df.columns]
    if not key_cols:
        idx = np.arange(len(df), dtype=np.uint64)
    else:
        key_df = df[key_cols].astype(str)
        idx = pd.util.hash_pandas_object(key_df, index=False).to_numpy(dtype=np.uint64)
    seed_mix = np.uint64((int(seed) * 11400714819323198485) % (2**64))
    mixed = idx ^ seed_mix
    denom = np.uint64(1_000_003)
    return (mixed % denom).astype(np.float64) / float(denom)


def _legacy_simulate(df: pd.DataFrame, exp: dict) -> tuple[pd.DataFrame, list[dict]]:
    alpha = exp["confblend"]["alpha"]
    tau_fixed = exp["setacsa"]["tau"]
    csa = exp["csaab"]
    params = CSAABParams(
        tau_min=csa["tau_min"],
        tau_max=csa["tau_max"],
        base_tau=csa["base_tau"],
        k_workload=csa["k_workload"],
        k_entropy=csa["k_entropy"],
        entropy_ref=csa["entropy_ref"],
        workload_ref=csa["workload_ref"],
    )

    p_u = _as_prob_matrix(df["p_u"])
    p_a = _as_prob_matrix(df["p_a"])

    g_u = p_u.argmax(axis=1).astype(int)
    g_a = p_a.argmax(axis=1).astype(int)
    c_u = p_u.max(axis=1).astype(float)
    H = np.apply_along_axis(entropy, 1, p_u)
    w = df["workload"].to_numpy(dtype=float) if "workload" in df.columns else np.full(len(df), 0.5, dtype=float)

    p_cb = _normalize_rows(alpha * p_u + (1.0 - alpha) * p_a)
    g_cb = p_cb.argmax(axis=1).astype(int)
    rng = np.random.default_rng(int(exp.get("seed", 1337)) + 9001)
    g_rand = rng.integers(0, p_u.shape[1], size=len(df), dtype=int)
    same = g_rand == g_u
    if np.any(same):
        g_rand[same] = (g_rand[same] + 1) % p_u.shape[1]

    tau_dyn = np.array([csaab_tau(params, wi, hi) for wi, hi in zip(w, H)], dtype=float)
    tau_dyn_entropy = np.array([csaab_tau(params, params.workload_ref, hi) for hi in H], dtype=float)
    tau_dyn_workload = np.array([csaab_tau(params, wi, params.entropy_ref) for wi in w], dtype=float)
    tau_set = np.full(len(df), float(tau_fixed), dtype=float)

    g_set = np.where(c_u >= tau_set, g_u, g_a)
    g_random_budget = np.where(c_u >= tau_set, g_u, g_rand)
    g_csaab = np.where(c_u >= tau_dyn, g_u, g_a)
    g_csaab_entropy = np.where(c_u >= tau_dyn_entropy, g_u, g_a)
    g_csaab_workload = np.where(c_u >= tau_dyn_workload, g_u, g_a)
    g_csaab_budget = g_csaab.copy()

    out = pd.DataFrame(
        {
            "g_u": g_u,
            "g_a": g_a,
            "c_u": c_u,
            "H_pu": H,
            "workload": w,
            "tau_set": tau_set,
            "tau_dyn": tau_dyn,
            "tau_dyn_budget": tau_dyn,
            "tau_dyn_entropy": tau_dyn_entropy,
            "tau_dyn_workload": tau_dyn_workload,
            "ali_set": np.array([ali_from_tau(v) for v in tau_set], dtype=float),
            "ali_dyn": np.array([ali_from_tau(v) for v in tau_dyn], dtype=float),
            "ali_dyn_budget": np.array([ali_from_tau(v) for v in tau_dyn], dtype=float),
            "g_hat_useronly": g_u,
            "g_hat_autoonly": g_a,
            "g_hat_confblend": g_cb,
            "g_hat_setacsa": g_set,
            "g_hat_random_budget": g_random_budget,
            "g_hat_csaab": g_csaab,
            "g_hat_csaab_budget": g_csaab_budget,
            "g_hat_csaab_entropy": g_csaab_entropy,
            "g_hat_csaab_workload": g_csaab_workload,
            "fold_id": np.full(len(df), -1, dtype=int),
        }
    )
    out = pd.concat([df[_select_id_columns(df)].reset_index(drop=True), out], axis=1)
    return out, []


def _quantile_tau_simulate(df: pd.DataFrame, exp: dict, *, seed: int, dataset_id: str) -> tuple[pd.DataFrame, list[dict]]:
    alpha = float(exp["confblend"]["alpha"])
    p_u = _as_prob_matrix(df["p_u"])
    p_a = _as_prob_matrix(df["p_a"])
    tau_cfg = exp.get("tau_calibration", {})

    g_u = p_u.argmax(axis=1).astype(int)
    g_a = p_a.argmax(axis=1).astype(int)
    c_u_raw = p_u.max(axis=1).astype(float)
    tie_break_eps = float(tau_cfg.get("tie_break_eps", 1e-6))
    if tie_break_eps > 0.0:
        jitter = _deterministic_tie_jitter(df, seed=seed)
        c_u = np.clip(c_u_raw + tie_break_eps * jitter, 0.0, 1.0)
    else:
        c_u = c_u_raw.copy()
    H = np.apply_along_axis(entropy, 1, p_u)
    w = df["workload"].to_numpy(dtype=float) if "workload" in df.columns else np.full(len(df), 0.5, dtype=float)

    p_cb = _normalize_rows(alpha * p_u + (1.0 - alpha) * p_a)
    g_cb = p_cb.argmax(axis=1).astype(int)
    rng = np.random.default_rng(int(seed) + 9001)
    g_rand = rng.integers(0, p_u.shape[1], size=len(df), dtype=int)
    same = g_rand == g_u
    if np.any(same):
        g_rand[same] = (g_rand[same] + 1) % p_u.shape[1]

    target_set = float(tau_cfg.get("target_gate_rate_set", tau_cfg.get("target_gate_rate", 0.10)))
    n_splits = int(tau_cfg.get("n_splits", 3))
    set_use_complement = bool(tau_cfg.get("set_use_complement_quantile", False))
    dyn_use_complement = bool(tau_cfg.get("dynamic_use_complement_quantile", True))

    dyn_cfg = tau_cfg.get("dynamic", {}) or {}
    dyn_params = DynamicGateRateParams(
        a=float(dyn_cfg.get("a", 0.0)),
        b_workload=float(dyn_cfg.get("b_workload", 2.5)),
        c_entropy=float(dyn_cfg.get("c_entropy", 2.5)),
        r_min=float(dyn_cfg.get("r_min", 0.05)),
        r_max=float(dyn_cfg.get("r_max", 0.30)),
    )

    subjects = df.get("subject_id", pd.Series(["S0"] * len(df))).astype(str).to_numpy()
    n_subjects = len(np.unique(subjects))
    n_splits = min(n_splits, max(2, n_subjects))
    cv = SubjectKFold(n_splits=n_splits, seed=seed)

    n = len(df)
    fold_id = np.full(n, -1, dtype=int)
    tau_set = np.full(n, np.nan, dtype=float)
    tau_dyn = np.full(n, np.nan, dtype=float)
    tau_dyn_budget = np.full(n, np.nan, dtype=float)
    tau_dyn_entropy = np.full(n, np.nan, dtype=float)
    tau_dyn_workload = np.full(n, np.nan, dtype=float)

    g_set = np.copy(g_u)
    g_random_budget = np.copy(g_u)
    g_csaab = np.copy(g_u)
    g_csaab_budget = np.copy(g_u)
    g_csaab_entropy = np.copy(g_u)
    g_csaab_workload = np.copy(g_u)

    tau_records: list[dict] = []

    for fid, (train_idx, test_idx) in enumerate(cv.split(subjects.tolist())):
        train_c = c_u[train_idx]
        tau_s = fit_tau_from_train_confidence(
            train_c,
            target_set,
            use_complement_quantile=set_use_complement,
        )
        achieved_set = float(np.mean(train_c < tau_s)) if len(train_c) else float("nan")

        trw = w[train_idx]
        trh = H[train_idx]

        tw = w[test_idx]
        th = H[test_idx]

        rate_both = target_gate_rate_dyn(tw, th, dyn_params)
        rate_h = target_gate_rate_dyn(
            np.zeros_like(tw),
            th,
            DynamicGateRateParams(
                a=dyn_params.a,
                b_workload=0.0,
                c_entropy=dyn_params.c_entropy,
                r_min=dyn_params.r_min,
                r_max=dyn_params.r_max,
            ),
        )
        rate_w = target_gate_rate_dyn(
            tw,
            np.zeros_like(th),
            DynamicGateRateParams(
                a=dyn_params.a,
                b_workload=dyn_params.b_workload,
                c_entropy=0.0,
                r_min=dyn_params.r_min,
                r_max=dyn_params.r_max,
            ),
        )

        q_both = 1.0 - rate_both if dyn_use_complement else rate_both
        q_h = 1.0 - rate_h if dyn_use_complement else rate_h
        q_w = 1.0 - rate_w if dyn_use_complement else rate_w

        budget_rates, budget_info = calibrate_budgeted_dynamic_rates(
            train_workload=trw,
            train_entropy=trh,
            test_workload=tw,
            test_entropy=th,
            params=dyn_params,
            target_mean_gate_rate=achieved_set,
        )
        # Budget-matched gating is defined on the realized gate/override budget
        # (gate event: c_u < tau), so use direct gate-rate quantiles.
        q_budget = budget_rates

        tau_both = np.nextafter(np.asarray(_quantile_linear(train_c, q=np.clip(q_both, 0.0, 1.0)), dtype=float), np.inf)
        tau_budget = np.nextafter(
            np.asarray(_quantile_linear(train_c, q=np.clip(q_budget, 0.0, 1.0)), dtype=float), np.inf
        )
        tau_h = np.nextafter(np.asarray(_quantile_linear(train_c, q=np.clip(q_h, 0.0, 1.0)), dtype=float), np.inf)
        tau_w = np.nextafter(np.asarray(_quantile_linear(train_c, q=np.clip(q_w, 0.0, 1.0)), dtype=float), np.inf)

        fold_id[test_idx] = fid
        tau_set[test_idx] = tau_s
        tau_dyn[test_idx] = tau_both
        tau_dyn_budget[test_idx] = tau_budget
        tau_dyn_entropy[test_idx] = tau_h
        tau_dyn_workload[test_idx] = tau_w

        g_set[test_idx] = np.where(c_u[test_idx] >= tau_s, g_u[test_idx], g_a[test_idx])
        g_random_budget[test_idx] = np.where(c_u[test_idx] >= tau_s, g_u[test_idx], g_rand[test_idx])
        g_csaab[test_idx] = np.where(c_u[test_idx] >= tau_both, g_u[test_idx], g_a[test_idx])
        g_csaab_budget[test_idx] = np.where(c_u[test_idx] >= tau_budget, g_u[test_idx], g_a[test_idx])
        g_csaab_entropy[test_idx] = np.where(c_u[test_idx] >= tau_h, g_u[test_idx], g_a[test_idx])
        g_csaab_workload[test_idx] = np.where(c_u[test_idx] >= tau_w, g_u[test_idx], g_a[test_idx])

        tau_records.append(
            {
                "dataset": dataset_id,
                "fold_id": int(fid),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "target_gate_rate_set": float(target_set),
                "tau_set": float(tau_s),
                "achieved_gate_rate_set_train": achieved_set,
                "train_conf_q01": float(np.quantile(train_c, 0.01)),
                "train_conf_q10": float(np.quantile(train_c, 0.10)),
                "train_conf_q50": float(np.quantile(train_c, 0.50)),
                "train_conf_q90": float(np.quantile(train_c, 0.90)),
                "train_conf_q99": float(np.quantile(train_c, 0.99)),
                "dyn_rate_mean_test": float(np.mean(rate_both)),
                "dyn_rate_min_test": float(np.min(rate_both)),
                "dyn_rate_max_test": float(np.max(rate_both)),
                "dyn_budget_rate_mean_test": float(np.mean(budget_rates)),
                "dyn_budget_shift": float(budget_info.get("shift", 0.0)),
                "dyn_budget_train_mean_after_shift": float(budget_info.get("train_budget_mean", float("nan"))),
                "gate_rate_set_test": float(np.mean(c_u[test_idx] < tau_s)),
                "gate_rate_dyn_test": float(np.mean(c_u[test_idx] < tau_both)),
                "gate_rate_dyn_budget_test": float(np.mean(c_u[test_idx] < tau_budget)),
            }
        )

    out = pd.DataFrame(
        {
            "g_u": g_u,
            "g_a": g_a,
            "c_u": c_u,
            "c_u_raw": c_u_raw,
            "H_pu": H,
            "workload": w,
            "tau_set": tau_set,
            "tau_dyn": tau_dyn,
            "tau_dyn_budget": tau_dyn_budget,
            "tau_dyn_entropy": tau_dyn_entropy,
            "tau_dyn_workload": tau_dyn_workload,
            "ali_set": np.array([ali_from_tau(v) for v in tau_set], dtype=float),
            "ali_dyn": np.array([ali_from_tau(v) for v in tau_dyn], dtype=float),
            "ali_dyn_budget": np.array([ali_from_tau(v) for v in tau_dyn_budget], dtype=float),
            "g_hat_useronly": g_u,
            "g_hat_autoonly": g_a,
            "g_hat_confblend": g_cb,
            "g_hat_setacsa": g_set,
            "g_hat_random_budget": g_random_budget,
            "g_hat_csaab": g_csaab,
            "g_hat_csaab_budget": g_csaab_budget,
            "g_hat_csaab_entropy": g_csaab_entropy,
            "g_hat_csaab_workload": g_csaab_workload,
            "fold_id": fold_id,
        }
    )
    out = pd.concat([df[_select_id_columns(df)].reset_index(drop=True), out], axis=1)
    return out, tau_records


def simulate_df(df: pd.DataFrame, exp: dict, *, seed: int, dataset_id: str) -> tuple[pd.DataFrame, list[dict]]:
    tau_cfg = exp.get("tau_calibration", {})
    enabled = bool(tau_cfg.get("enabled", False))
    if enabled:
        return _quantile_tau_simulate(df, exp, seed=seed, dataset_id=dataset_id)
    return _legacy_simulate(df, exp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--datasets_yaml", type=Path, required=True)
    ap.add_argument("--exp_yaml", type=Path, required=True)
    ap.add_argument("--paper_mode", type=int, choices=[0, 1], default=int(os.environ.get("PAPER_MODE", "0")))
    args = ap.parse_args()

    dsreg = load_yaml(args.datasets_yaml)["datasets"]
    exp = load_yaml(args.exp_yaml)["experiment"]

    data_root = Path(os.environ.get("DATA_ROOT", "lake"))
    derived_root = data_root / "derived"
    exp_id = os.environ.get("EXP_ID", exp["id"])
    results_root = Path("results") / exp_id
    results_root.mkdir(parents=True, exist_ok=True)

    seed = int(exp.get("seed", 1337))
    n_simulated = 0

    for ds_id, info in dsreg.items():
        if int(info.get("tier", 99)) > args.tier:
            continue
        parquet_path = derived_root / ds_id / info.get("version", "unknown") / "windows.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        paper_mode = int(args.paper_mode) == 1
        role = str(info.get("arbitration_role", "full")).strip().lower() or "full"
        if paper_mode:
            if role in {"p_a_only", "workload_only"} and os.environ.get("INCLUDE_NONCLAIM_DATASETS_IN_PAPER_MODE", "0") != "1":
                print(f"[SIM] {ds_id}: skipped (dataset_role={role}, excluded from e2e arbitration claims)")
                continue
            pred_root = results_root / ds_id / "PRED"
            pu_path = pred_root / "pu_pred.parquet"
            pa_path = pred_root / "pa_pred.parquet"
            df = _merge_predicted_priors(df, pu_path, pa_path)
        else:
            if "p_u" not in df.columns or "p_a" not in df.columns:
                continue

        sim, tau_records = simulate_df(df, exp, seed=seed, dataset_id=ds_id)

        out_dir = results_root / ds_id
        out_dir.mkdir(parents=True, exist_ok=True)
        sim.to_parquet(out_dir / "sim.parquet", index=False)
        if tau_records:
            (out_dir / "tau_calibration.json").write_text(json.dumps(tau_records, indent=2), encoding="utf-8")
        print(f"[SIM] {ds_id}: {len(sim)} windows")
        n_simulated += 1

    if int(args.paper_mode) == 1 and n_simulated == 0:
        raise SystemExit("paper_mode=1 produced no simulated datasets (fail-closed).")


if __name__ == "__main__":
    main()
