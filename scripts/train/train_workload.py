from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import yaml

from command_departure_benchmark.eval.splits import SubjectKFold


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _subject_train_val_split(train_idx: np.ndarray, subjects: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects[train_idx])
    rng.shuffle(uniq)
    n_val = max(1, int(round(0.2 * len(uniq)))) if len(uniq) > 1 else 1
    val_subs = set(uniq[:n_val].tolist())
    val_mask = np.isin(subjects[train_idx], list(val_subs))
    return train_idx[~val_mask], train_idx[val_mask]


def _as_float(series: pd.Series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=np.float32)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _prepare(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if "workload" not in df.columns:
        raise ValueError("missing workload column")
    if "pupil_metric" in df.columns:
        pupil = _as_float(df["pupil_metric"], default=0.0)
    else:
        pupil_cols = [c for c in df.columns if "pupil" in c.lower()]
        if not pupil_cols:
            raise ValueError("missing pupil metric columns")
        pupil = _as_float(df[pupil_cols].mean(axis=1), default=0.0)
    illum = _as_float(df["illuminance"], default=0.0) if "illuminance" in df.columns else np.zeros(len(df), dtype=np.float32)
    y = _as_float(df["workload"], default=0.5)
    subjects = df.get("subject_id", pd.Series(["S0"] * len(df))).astype(str).to_numpy()
    X_base = pupil.reshape(-1, 1)
    X_conf = np.stack([pupil, illum], axis=1)
    return X_base, X_conf, y, subjects


def run_dataset(df: pd.DataFrame, *, seed: int, max_folds: int) -> tuple[list[dict[str, Any]], dict[str, Any], Ridge, Ridge]:
    X_base, X_conf, y, subjects = _prepare(df)
    n_subjects = len(np.unique(subjects))
    n_splits = min(max_folds, max(2, n_subjects))
    cv = SubjectKFold(n_splits=n_splits, seed=seed)

    fold_rows = []
    models_base = []
    models_conf = []
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(subjects.tolist())):
        tr_idx, _ = _subject_train_val_split(train_idx, subjects, seed + fold_id)

        m_base = Ridge(alpha=1.0)
        m_base.fit(X_base[tr_idx], y[tr_idx])
        pred_base = m_base.predict(X_base[test_idx])
        met_base = _metrics(y[test_idx], pred_base)
        fold_rows.append({"fold": fold_id, "model": "pupil_only", **met_base})
        models_base.append(m_base)

        m_conf = Ridge(alpha=1.0)
        m_conf.fit(X_conf[tr_idx], y[tr_idx])
        pred_conf = m_conf.predict(X_conf[test_idx])
        met_conf = _metrics(y[test_idx], pred_conf)
        fold_rows.append({"fold": fold_id, "model": "pupil_plus_illuminance", **met_conf})
        models_conf.append(m_conf)

    fold_df = pd.DataFrame(fold_rows)
    agg_df = (
        fold_df.groupby("model", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index(drop=True)
    )
    model_rows = [
        {
            "model": str(row["model"]),
            "mae_mean": float(row["mae_mean"]),
            "mae_std": float(row["mae_std"]),
            "r2_mean": float(row["r2_mean"]),
            "r2_std": float(row["r2_std"]),
        }
        for _, row in agg_df.iterrows()
    ]
    metrics = {
        "n_rows": int(len(df)),
        "n_subjects": int(n_subjects),
        "models": model_rows,
    }

    # Return last fold models for transfer benchmarking.
    return fold_rows, metrics, models_base[-1], models_conf[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--datasets_yaml", type=Path, required=True)
    ap.add_argument("--exp_yaml", type=Path, required=True)
    ap.add_argument("--max_folds", type=int, default=5)
    args = ap.parse_args()

    dsreg = load_yaml(args.datasets_yaml)["datasets"]
    exp = load_yaml(args.exp_yaml)["experiment"]
    exp_id = os.environ.get("EXP_ID", exp["id"])
    seed = int(exp.get("seed", 1337))

    data_root = Path(os.environ.get("DATA_ROOT", "lake"))
    derived_root = data_root / "derived"
    results_root = Path("results") / exp_id

    # Per-dataset training.
    prepared: dict[str, dict[str, Any]] = {}
    for ds_id, info in dsreg.items():
        if int(info.get("tier", 99)) > args.tier:
            continue
        windows = derived_root / ds_id / info.get("version", "unknown") / "windows.parquet"
        if not windows.exists():
            continue
        df = pd.read_parquet(windows)
        out_dir = results_root / ds_id
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            fold_rows, metrics, model_base, model_conf = run_dataset(df, seed=seed, max_folds=args.max_folds)
        except Exception as exc:
            print(f"[TRAIN_W] {ds_id}: skipped ({exc})")
            continue
        (out_dir / "workload_fold_metrics.json").write_text(json.dumps(fold_rows, indent=2))
        (out_dir / "workload_metrics.json").write_text(json.dumps(metrics, indent=2))
        prepared[ds_id] = {"df": df, "base": model_base, "conf": model_conf}
        print(f"[TRAIN_W] {ds_id}: wrote workload metrics")

    # Cross-dataset transfer.
    transfer_rows = []
    for src_id, src_obj in prepared.items():
        for dst_id, dst_obj in prepared.items():
            if src_id == dst_id:
                continue
            try:
                Xb, Xc, y_dst, _ = _prepare(dst_obj["df"])
            except Exception:
                continue
            pred_b = src_obj["base"].predict(Xb)
            pred_c = src_obj["conf"].predict(Xc)
            transfer_rows.append(
                {
                    "train_dataset": src_id,
                    "test_dataset": dst_id,
                    "model": "pupil_only",
                    **_metrics(y_dst, pred_b),
                }
            )
            transfer_rows.append(
                {
                    "train_dataset": src_id,
                    "test_dataset": dst_id,
                    "model": "pupil_plus_illuminance",
                    **_metrics(y_dst, pred_c),
                }
            )

    if transfer_rows:
        transfer_path = results_root / "workload_transfer.json"
        transfer_path.write_text(json.dumps(transfer_rows, indent=2))
        print(f"[TRAIN_W] wrote transfer report: {transfer_path}")


if __name__ == "__main__":
    main()
