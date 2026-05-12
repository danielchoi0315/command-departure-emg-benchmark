from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from command_departure_benchmark.eval.metrics import accuracy
from command_departure_benchmark.eval.splits import SubjectKFold


POPULATION_MODEL_DATASETS = {
    "ninapro_db10_meganepro",
    "physionet_hyser",
    "physionet_grabmyo",
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def _global_prior(train_df: pd.DataFrame, y_col: str, n_classes: int) -> np.ndarray:
    counts = np.bincount(train_df[y_col].astype(int).to_numpy(), minlength=n_classes).astype(float)
    counts += 1e-3
    return counts / counts.sum()


def _align_proba_to_global_classes(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    aligned = np.full((proba.shape[0], n_classes), 1e-12, dtype=float)
    cls = np.asarray(classes, dtype=int)
    valid = (cls >= 0) & (cls < n_classes)
    cls = cls[valid]
    if cls.size > 0:
        aligned[:, cls] = proba[:, valid]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _as_vec(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).reshape(-1)
    if pd.isna(value):
        return np.asarray([], dtype=float)
    return np.asarray([float(value)], dtype=float)


def _feature_matrix(df: pd.DataFrame, *, preferred: str = "X_pa") -> np.ndarray:
    col = preferred if preferred in df.columns else "X_pu"
    if col not in df.columns:
        return np.zeros((len(df), 0), dtype=float)
    vecs = [_as_vec(v) for v in df[col]]
    if not vecs:
        return np.zeros((0, 0), dtype=float)
    max_dim = max((v.size for v in vecs), default=0)
    if max_dim <= 0:
        return np.zeros((len(vecs), 0), dtype=float)
    out = np.zeros((len(vecs), max_dim), dtype=float)
    for i, v in enumerate(vecs):
        d = min(max_dim, v.size)
        if d > 0:
            out[i, :d] = v[:d]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _harmonic_goal_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(r"p_goal_\d+", str(c).lower())]
    if not cols:
        return []
    return sorted(cols, key=lambda c: int(re.search(r"(\d+)", c).group(1)))


def _harmonic_goal_probs(
    test_df: pd.DataFrame,
    *,
    goal_cols: list[str],
    default: np.ndarray,
) -> tuple[np.ndarray, float]:
    out = []
    n_nonzero = 0
    for _, row in test_df.iterrows():
        vec = pd.to_numeric(row[goal_cols], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        s = float(vec.sum())
        if s > 0:
            out.append(vec / s)
            n_nonzero += 1
        else:
            out.append(default)
    arr = np.vstack(out) if out else np.zeros((0, default.size), dtype=float)
    coverage = float(n_nonzero / max(1, len(test_df)))
    return arr, coverage


def _id_cols(df: pd.DataFrame) -> list[str]:
    preferred = ["dataset", "subject_id", "session_id", "trial_id", "window_id", "g_star"]
    return [c for c in preferred if c in df.columns]


def _run_population_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    n_classes: int,
    max_train_rows: int,
    seed: int,
    model_kind: str,
) -> tuple[np.ndarray, str]:
    x_train = _feature_matrix(train_df, preferred="X_pa")
    x_test = _feature_matrix(test_df, preferred="X_pa")
    y_train = train_df["g_star"].astype(int).to_numpy()

    if x_train.shape[1] == 0 or len(np.unique(y_train)) < 2 or len(train_df) < 20:
        return np.tile(_global_prior(train_df, y_col="g_star", n_classes=n_classes), (len(test_df), 1)), "fold_global_frequency_prior"

    if max_train_rows > 0 and len(train_df) > max_train_rows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(np.arange(len(train_df)), size=max_train_rows, replace=False)
        keep = np.sort(keep)
        x_train = x_train[keep]
        y_train = y_train[keep]

    if model_kind == "gaussian_nb":
        clf = GaussianNB()
        method = "population_xpu_gaussian_nb"
    else:
        # scikit-learn 1.7+ deprecates/removes the explicit multi_class arg.
        try:
            clf = LogisticRegression(max_iter=1500, solver="lbfgs", multi_class="multinomial")
        except TypeError:
            clf = LogisticRegression(max_iter=1500, solver="lbfgs")
        method = "population_xpu_logreg"
    clf.fit(x_train, y_train)
    probs = _align_proba_to_global_classes(clf.predict_proba(x_test), clf.classes_, n_classes)
    # Keep p_a as a softer autonomy prior and avoid degenerate over-confidence.
    prior = _global_prior(train_df, y_col="g_star", n_classes=n_classes).reshape(1, -1)
    probs = 0.7 * probs + 0.3 * np.repeat(prior, len(test_df), axis=0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, method


def _run_dataset(
    ds_id: str,
    df: pd.DataFrame,
    *,
    seed: int,
    max_folds: int,
    max_train_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    y = df["g_star"].astype(int).to_numpy()
    subjects = df.get("subject_id", pd.Series(["S0"] * len(df))).astype(str).to_numpy()
    n_classes = int(y.max()) + 1
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 2:
        raise ValueError(f"{ds_id}: requires >=2 subjects for subject-wise p_a splits")

    n_splits = min(max_folds, max(2, len(unique_subjects)))
    cv = SubjectKFold(n_splits=n_splits, seed=seed)

    fold_metrics: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []
    ids = _id_cols(df)

    goal_cols = _harmonic_goal_cols(df) if ds_id == "harmonic" else []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(subjects.tolist())):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        global_prior = _global_prior(train_df, y_col="g_star", n_classes=n_classes)

        coverage = 1.0
        if ds_id == "harmonic" and goal_cols:
            probs, coverage = _harmonic_goal_probs(test_df, goal_cols=goal_cols, default=global_prior)
            method = "harmonic_real_goal_prior"
        elif ds_id in POPULATION_MODEL_DATASETS:
            model_kind = "gaussian_nb" if ds_id in {"physionet_hyser", "physionet_grabmyo"} else "logreg"
            probs, method = _run_population_model(
                train_df,
                test_df,
                n_classes=n_classes,
                max_train_rows=max_train_rows,
                seed=seed + fold_id,
                model_kind=model_kind,
            )
        else:
            probs = np.tile(global_prior.reshape(1, -1), (len(test_df), 1))
            method = "fold_global_frequency_prior"

        y_test = test_df["g_star"].astype(int).to_numpy()
        y_pred = probs.argmax(axis=1) if len(probs) else np.zeros(0, dtype=int)
        fold_metrics.append(
            {
                "fold": fold_id,
                "method": method,
                "pa_accuracy": accuracy(y_test, y_pred),
                "pa_nll": _nll(probs, y_test),
                "coverage": float(coverage),
                "n_train_rows": int(len(train_df)),
                "n_test_rows": int(len(test_df)),
                "n_train_subjects": int(train_df["subject_id"].astype(str).nunique()) if "subject_id" in train_df.columns else 0,
                "n_test_subjects": int(test_df["subject_id"].astype(str).nunique()) if "subject_id" in test_df.columns else 0,
            }
        )
        mappings.append(
            {
                "fold": fold_id,
                "method": method,
                "goal_cols": goal_cols,
                "global_prior": global_prior.astype(float).tolist(),
            }
        )

        part = test_df[ids].reset_index(drop=True).copy()
        part["fold_id"] = int(fold_id)
        part["method"] = method
        part["p_a_pred"] = [row.astype(float).tolist() for row in probs]
        pred_parts.append(part)

    if not pred_parts:
        raise ValueError(f"{ds_id}: no fold predictions produced")

    pred_df = pd.concat(pred_parts, ignore_index=True)
    if ids and not pred_df.empty:
        if pred_df.duplicated(subset=ids).any():
            dup_n = int(pred_df.duplicated(subset=ids).sum())
            raise ValueError(f"{ds_id}: duplicate pa_pred ids detected (n={dup_n})")
        if len(pred_df) != len(df):
            raise ValueError(f"{ds_id}: pa_pred coverage mismatch rows={len(pred_df)} expected={len(df)}")

    return fold_metrics, mappings, pred_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--datasets_yaml", type=Path, required=True)
    ap.add_argument("--exp_yaml", type=Path, required=True)
    ap.add_argument("--max_folds", type=int, default=5)
    ap.add_argument("--max_train_rows", type=int, default=int(os.environ.get("PA_MAX_TRAIN_ROWS", "250000")))
    args = ap.parse_args()

    dsreg = load_yaml(args.datasets_yaml)["datasets"]
    exp = load_yaml(args.exp_yaml)["experiment"]
    exp_id = os.environ.get("EXP_ID", exp["id"])
    seed = int(exp.get("seed", 1337))

    data_root = Path(os.environ.get("DATA_ROOT", "lake"))
    derived_root = data_root / "derived"
    results_root = Path("results") / exp_id

    for ds_id, info in dsreg.items():
        if int(info.get("tier", 99)) > args.tier:
            continue
        windows = derived_root / ds_id / info.get("version", "unknown") / "windows.parquet"
        if not windows.exists():
            continue
        df = pd.read_parquet(windows)
        if "g_star" not in df.columns:
            continue

        out_dir = results_root / ds_id
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            fold_metrics, mappings, pred_df = _run_dataset(
                ds_id,
                df,
                seed=seed,
                max_folds=args.max_folds,
                max_train_rows=args.max_train_rows,
            )
        except Exception as exc:
            print(f"[TRAIN_PA] {ds_id}: skipped ({exc})")
            continue

        fm_df = pd.DataFrame(fold_metrics)
        agg = fm_df[["pa_accuracy", "pa_nll", "coverage"]].agg(["mean", "std"]).to_dict() if not fm_df.empty else {}
        metrics = {
            "dataset": ds_id,
            "summary": agg,
            "n_rows": int(len(df)),
            "methods": sorted(set(fm_df["method"].astype(str).tolist())) if not fm_df.empty else [],
        }

        pred_dir = out_dir / "PRED"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(pred_dir / "pa_pred.parquet", index=False)

        (out_dir / "pa_fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2))
        (out_dir / "pa_mappings.json").write_text(json.dumps(mappings, indent=2))
        (out_dir / "pa_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"[TRAIN_PA] {ds_id}: wrote pa metrics + {pred_dir / 'pa_pred.parquet'}")


if __name__ == "__main__":
    main()
