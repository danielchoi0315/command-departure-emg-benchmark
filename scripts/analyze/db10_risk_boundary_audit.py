#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LEAKAGE_EXCLUDED_METADATA = {
    "label",
    "label_raw",
    "source_path",
    "record_id",
    "episode_id",
    "trial_id",
    "window_id",
    "subject_id",
    "grasp_repetition",
}

SAFE_METADATA_COLS = [
    "position_id",
    "object_id",
    "object_part",
    "dynamic_flag",
    "gaze_available",
    "pupil_left_available",
    "pupil_right_available",
    "acc_available",
    "gyr_available",
    "assist_context_mode",
]

USER_MODELS = ["lda", "logreg", "random_forest", "extra_trees"]
TAU_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
CONF_GRID_SIZE = 41
MIX_LAMBDAS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
SHIFT_WINDOWS = [0, 1, 3, 5, 10, 20]


@dataclass(frozen=True)
class Fold:
    split: str
    fold_id: int
    train_idx: np.ndarray
    dev_idx: np.ndarray
    test_idx: np.ndarray
    train_subjects: tuple[str, ...]
    dev_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.nanmax(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)


def clip_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def align_proba(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.full((len(proba), n_classes), 1e-12, dtype=float)
    cls = np.asarray(classes, dtype=int)
    valid = (cls >= 0) & (cls < n_classes)
    if valid.any():
        out[:, cls[valid]] = proba[:, valid]
    return clip_prob(out)


def class_prior(y: np.ndarray, n_classes: int, alpha: float = 1.0) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(float)
    counts += float(alpha)
    return counts / counts.sum()


def active_macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    active = np.asarray(y) != 0
    if not active.any():
        return float("nan")
    return float(f1_score(np.asarray(y)[active], np.asarray(pred)[active], average="macro", zero_division=0))


def ece_score(y: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    prob = clip_prob(prob)
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    y = np.asarray(y, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float((pred[mask] == y[mask]).mean()) - float(conf[mask].mean()))
    return float(ece)


def brier_score(y: np.ndarray, prob: np.ndarray) -> float:
    prob = clip_prob(prob)
    eye = np.eye(prob.shape[1], dtype=float)[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((prob - eye) ** 2, axis=1)))


def risk_coverage_auc(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(y) == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    err = (pred[order] != y[order]).astype(float)
    coverage = np.arange(1, len(err) + 1, dtype=float) / len(err)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1, dtype=float)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(risk, coverage))


def prediction_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    prob = clip_prob(prob)
    pred = prob.argmax(axis=1).astype(int)
    labels = np.arange(prob.shape[1])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "active_macro_f1": active_macro_f1(y, pred),
        "nll": float(log_loss(y, prob, labels=labels)),
        "ece": ece_score(y, prob),
        "brier": brier_score(y, prob),
    }


def mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, mean, mean
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def subject_split(subjects: np.ndarray, seed: int, frac_dev: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    uniq = np.array(sorted(pd.Series(subjects).astype(str).unique()))
    if len(uniq) < 3:
        return np.arange(len(subjects)), np.asarray([], dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_dev = max(1, int(round(len(uniq) * frac_dev)))
    dev_sub = set(uniq[:n_dev].tolist())
    dev = np.where(np.isin(subjects.astype(str), list(dev_sub)))[0]
    train = np.where(~np.isin(subjects.astype(str), list(dev_sub)))[0]
    return train.astype(int), dev.astype(int)


def make_folds(meta: pd.DataFrame, split: str, seed: int, n_splits: int) -> list[Fold]:
    subjects = meta["subject_id"].astype(str).to_numpy()
    groups = meta.get("group", pd.Series(["all"] * len(meta))).astype(str).to_numpy()
    rows: list[Fold] = []

    if split == "all_subject_kfold":
        uniq = np.array(sorted(pd.Series(subjects).unique()))
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        chunks = np.array_split(uniq, min(n_splits, len(uniq)))
        for fid, test_sub in enumerate(chunks):
            test_mask = np.isin(subjects, test_sub.astype(str))
            train_outer = np.where(~test_mask)[0]
            core_rel, dev_rel = subject_split(subjects[train_outer], seed + fid)
            train_idx = train_outer[core_rel]
            dev_idx = train_outer[dev_rel]
            test_idx = np.where(test_mask)[0]
            rows.append(make_fold(split, fid, train_idx, dev_idx, test_idx, subjects))
    elif split == "amputee_loso":
        amp_subs = np.array(sorted(pd.Series(subjects[groups == "amputee"]).unique()))
        for fid, test_sub in enumerate(amp_subs):
            test_mask = subjects == test_sub
            train_outer = np.where((groups == "amputee") & (~test_mask))[0]
            core_rel, dev_rel = subject_split(subjects[train_outer], seed + fid)
            rows.append(make_fold(split, fid, train_outer[core_rel], train_outer[dev_rel], np.where(test_mask)[0], subjects))
    elif split == "able_to_amputee":
        train_outer = np.where(groups == "able_bodied")[0]
        test_idx = np.where(groups == "amputee")[0]
        core_rel, dev_rel = subject_split(subjects[train_outer], seed)
        rows.append(make_fold(split, 0, train_outer[core_rel], train_outer[dev_rel], test_idx, subjects))
    else:
        raise ValueError(f"unknown split: {split}")
    return [f for f in rows if len(f.train_idx) and len(f.test_idx)]


def make_fold(split: str, fid: int, train_idx: np.ndarray, dev_idx: np.ndarray, test_idx: np.ndarray, subjects: np.ndarray) -> Fold:
    return Fold(
        split=split,
        fold_id=int(fid),
        train_idx=np.asarray(train_idx, dtype=int),
        dev_idx=np.asarray(dev_idx, dtype=int),
        test_idx=np.asarray(test_idx, dtype=int),
        train_subjects=tuple(sorted(pd.Series(subjects[train_idx]).astype(str).unique())),
        dev_subjects=tuple(sorted(pd.Series(subjects[dev_idx]).astype(str).unique())),
        test_subjects=tuple(sorted(pd.Series(subjects[test_idx]).astype(str).unique())),
    )


def fit_predict_classifier(
    kind: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    n_classes: int,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    if x_train.shape[1] == 0 or len(np.unique(y_train)) < 2:
        prior = class_prior(y_train, n_classes)
        return np.repeat(prior[None, :], len(x_eval), axis=0)

    if kind == "lda":
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    elif kind == "logreg":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=seed),
        )
    elif kind == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=seed,
        )
    elif kind == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=240,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown classifier kind: {kind}")

    clf.fit(x_train, y_train)
    classes = clf.classes_ if hasattr(clf, "classes_") else clf[-1].classes_
    return align_proba(clf.predict_proba(x_eval), classes, n_classes)


def metadata_matrix(meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in SAFE_METADATA_COLS if c in meta.columns and c not in LEAKAGE_EXCLUDED_METADATA]
    if not cols:
        return np.zeros((len(meta), 0), dtype=float), []
    block = meta[cols].copy()
    for c in block.columns:
        block[c] = block[c].astype(str).fillna("__missing__")
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return enc.fit_transform(block).astype(np.float32), cols


def conditioned_prior_predict(
    train_meta: pd.DataFrame,
    train_y: np.ndarray,
    test_meta: pd.DataFrame,
    *,
    cols: list[str],
    n_classes: int,
    alpha: float = 1.0,
    active_only: bool = False,
) -> np.ndarray:
    train_y = np.asarray(train_y, dtype=int)
    if active_only:
        active = train_y != 0
        train_meta = train_meta.loc[active].copy()
        train_y = train_y[active]
    global_prior = class_prior(train_y, n_classes, alpha=alpha)
    if active_only and n_classes > 1:
        global_prior[0] = 1e-12
        global_prior = global_prior / global_prior.sum()
    if not cols:
        return np.repeat(global_prior[None, :], len(test_meta), axis=0)
    train_key = train_meta[cols].astype(str).agg("||".join, axis=1)
    test_key = test_meta[cols].astype(str).agg("||".join, axis=1)
    counts: dict[str, np.ndarray] = {}
    for key, y in zip(train_key, train_y):
        if key not in counts:
            counts[key] = np.full(n_classes, float(alpha), dtype=float)
            if active_only and n_classes > 1:
                counts[key][0] = 1e-12
        counts[key][int(y)] += 1.0
    out = np.empty((len(test_meta), n_classes), dtype=float)
    for i, key in enumerate(test_key):
        if key in counts:
            vec = counts[key]
            out[i] = vec / vec.sum()
        else:
            out[i] = global_prior
    return clip_prob(out)


def apply_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    prob = clip_prob(prob)
    return softmax(np.log(prob) / max(float(temperature), 1e-6))


def calibrate_temperature(dev_prob: np.ndarray, dev_y: np.ndarray, test_prob: np.ndarray) -> tuple[np.ndarray, float]:
    dev_prob = clip_prob(dev_prob)
    labels = np.arange(dev_prob.shape[1])

    def obj(t: float) -> float:
        p = apply_temperature(dev_prob, t)
        return float(log_loss(dev_y, p, labels=labels))

    res = minimize_scalar(obj, bounds=(0.05, 10.0), method="bounded", options={"xatol": 1e-3})
    t = float(res.x) if res.success else 1.0
    return apply_temperature(test_prob, t), t


def calibrate_dirichlet(dev_prob: np.ndarray, dev_y: np.ndarray, test_prob: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    dev_x = np.log(clip_prob(dev_prob))
    test_x = np.log(clip_prob(test_prob))
    if len(np.unique(dev_y)) < 2:
        return clip_prob(test_prob), "insufficient_dev_classes"
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=None, random_state=seed)
    clf.fit(dev_x, dev_y)
    return align_proba(clf.predict_proba(test_x), clf.classes_, test_prob.shape[1]), "dirichlet_logreg"


def degrade_prob(
    prob: np.ndarray,
    meta: pd.DataFrame,
    condition: str,
    *,
    seed: int,
    n_classes: int,
    lam: float = 0.0,
    shift: int = 0,
) -> np.ndarray:
    prob = clip_prob(prob)
    rng = np.random.default_rng(seed)
    if condition == "clean":
        return prob
    if condition == "uniform_mix":
        uniform = np.full_like(prob, 1.0 / n_classes)
        return clip_prob((1.0 - lam) * prob + lam * uniform)
    if condition == "global_shuffle":
        return prob[rng.permutation(len(prob))]
    if condition == "within_subject_shuffle":
        out = prob.copy()
        for _, idx in meta.groupby("subject_id", sort=False).indices.items():
            ii = np.asarray(idx, dtype=int)
            out[ii] = out[ii[rng.permutation(len(ii))]]
        return out
    if condition == "trial_shift":
        if shift <= 0:
            return prob
        out = prob.copy()
        sort_cols = [c for c in ["subject_id", "episode_id", "prefix_time_s", "timestamp_s"] if c in meta.columns]
        ordered = meta.reset_index().sort_values(sort_cols, kind="stable") if sort_cols else meta.reset_index()
        for _, block in ordered.groupby(["subject_id", "episode_id"], sort=False):
            idx = block["index"].to_numpy(dtype=int)
            if len(idx) <= 1:
                continue
            src = np.maximum(np.arange(len(idx)) - shift, 0)
            out[idx] = prob[idx[src]]
        return out
    raise ValueError(f"unknown degradation condition: {condition}")


def cd_margin_policy(p_u: np.ndarray, p_a: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_u = clip_prob(p_u)
    p_a = clip_prob(p_a)
    g_u = p_u.argmax(axis=1).astype(int)
    g_a = p_a.argmax(axis=1).astype(int)
    cdc = np.maximum(0.0, p_u[np.arange(len(g_u)), g_u] - p_u[np.arange(len(g_a)), g_a])
    take_assist = (g_u != g_a) & (cdc <= float(tau))
    pred = g_u.copy()
    pred[take_assist] = g_a[take_assist]
    score = p_u[np.arange(len(g_u)), g_u].copy()
    score[take_assist] = p_a[np.arange(len(g_a)), g_a][take_assist]
    return pred, take_assist, score


def conf_gate_policy(p_u: np.ndarray, p_a: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_u = clip_prob(p_u)
    p_a = clip_prob(p_a)
    g_u = p_u.argmax(axis=1).astype(int)
    g_a = p_a.argmax(axis=1).astype(int)
    c_a = p_a.max(axis=1)
    take_assist = (g_u != g_a) & (c_a >= float(threshold))
    pred = g_u.copy()
    pred[take_assist] = g_a[take_assist]
    score = p_u[np.arange(len(g_u)), g_u].copy()
    score[take_assist] = c_a[take_assist]
    return pred, take_assist, score


def policy_metrics(y: np.ndarray, p_u: np.ndarray, p_a: np.ndarray, pred: np.ndarray, take_assist: np.ndarray, score: np.ndarray) -> dict[str, float]:
    g_u = clip_prob(p_u).argmax(axis=1).astype(int)
    g_a = clip_prob(p_a).argmax(axis=1).astype(int)
    cdc = np.maximum(0.0, p_u[np.arange(len(g_u)), g_u] - p_u[np.arange(len(g_a)), g_a])
    policy_cdc = cdc * np.asarray(take_assist, dtype=float)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "active_macro_f1": active_macro_f1(y, pred),
        "risk_coverage_auc": risk_coverage_auc(y, pred, score),
        "mean_cdc": float(np.mean(cdc[take_assist])) if np.any(take_assist) else 0.0,
        "expected_cdc": float(np.mean(policy_cdc)),
        "mean_cdc_all": float(np.mean(policy_cdc)),
        "candidate_cdc_all": float(np.mean(cdc)),
        "intervention_rate": float(np.mean(take_assist)),
        "changed_vs_user_rate": float(np.mean(pred != g_u)),
    }


def select_dev_threshold(dev_y: np.ndarray, dev_pu: np.ndarray, dev_pa: np.ndarray, metric: str, budget: float) -> float:
    c_a = clip_prob(dev_pa).max(axis=1)
    vals = np.unique(np.quantile(c_a, np.linspace(0, 1, CONF_GRID_SIZE)))
    vals = np.unique(np.concatenate([[float("inf")], vals, [np.nextafter(c_a.min(), -np.inf)]]))
    best_key: tuple[float, float] | None = None
    best_thr = float("inf")
    for thr in vals:
        pred, take, score = conf_gate_policy(dev_pu, dev_pa, float(thr))
        pm = policy_metrics(dev_y, dev_pu, dev_pa, pred, take, score)
        if metric == "cdc" and pm["expected_cdc"] > budget + 1e-12:
            continue
        if metric == "intervention" and pm["intervention_rate"] > budget + 1e-12:
            continue
        key = (pm["active_macro_f1"], -pm["expected_cdc"], -pm["intervention_rate"])
        if best_key is None or key > best_key:
            best_key = key
            best_thr = float(thr)
    return best_thr


def nondominated(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return points.copy()
    cdc_col = "expected_cdc" if "expected_cdc" in points.columns else "mean_cdc_all"
    arr = points[["active_macro_f1", cdc_col, "intervention_rate"]].to_numpy(dtype=float)
    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            better_or_equal = arr[j, 0] >= arr[i, 0] and arr[j, 1] <= arr[i, 1] and arr[j, 2] <= arr[i, 2]
            strictly = arr[j, 0] > arr[i, 0] or arr[j, 1] < arr[i, 1] or arr[j, 2] < arr[i, 2]
            if better_or_equal and strictly:
                keep[i] = False
                break
    return points.loc[keep].copy()


def run(args: argparse.Namespace) -> None:
    processed_root = args.processed_root / "db10"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(processed_root / "metadata.csv")
    y = np.load(processed_root / "labels.npy", mmap_mode="r").astype(int)
    user = np.load(processed_root / "user_features.npy", mmap_mode="r")
    assist = np.load(processed_root / "assist_features.npy", mmap_mode="r")
    n_classes = int(max(int(np.max(y)), len(np.load(processed_root / "label_vocab.npy", allow_pickle=True)) - 1)) + 1
    if len(meta) != len(y) or len(user) != len(y) or len(assist) != len(y):
        raise SystemExit("DB10 metadata/features/labels row counts do not match")
    if "label" in meta.columns and not np.array_equal(meta["label"].to_numpy(dtype=int), np.asarray(y, dtype=int)):
        raise SystemExit("DB10 metadata label column does not match labels.npy; row alignment is unsafe")
    if "subject_id" not in meta.columns:
        raise SystemExit("DB10 metadata is missing subject_id; subject-disjoint splits cannot be verified")
    if meta["subject_id"].isna().any():
        raise SystemExit("DB10 metadata contains missing subject_id values")
    if assist.shape[1] < user.shape[1]:
        raise SystemExit("assistive features must have at least user_feature_dim columns for prefix/tail controls")

    meta_x, meta_cols = metadata_matrix(meta)
    user_dim = int(user.shape[1])
    assist_prefix = np.asarray(assist[:, :user_dim], dtype=np.float32)
    assist_tail = np.asarray(assist[:, user_dim:], dtype=np.float32)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    folds: list[Fold] = []
    for split in splits:
        folds.extend(make_folds(meta, split, seed=args.seed + sum(ord(c) for c in split), n_splits=args.n_splits))
    if args.max_folds and args.max_folds > 0:
        folds = folds[: args.max_folds]

    fold_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for fold in folds:
        print(f"[DB10_AUDIT] split={fold.split} fold={fold.fold_id} train={len(fold.train_idx)} dev={len(fold.dev_idx)} test={len(fold.test_idx)}", flush=True)
        tr, dev, te = fold.train_idx, fold.dev_idx, fold.test_idx
        y_tr, y_dev, y_te = y[tr], y[dev] if len(dev) else y[tr], y[te]
        meta_tr, meta_dev, meta_te = meta.iloc[tr].reset_index(drop=True), meta.iloc[dev].reset_index(drop=True), meta.iloc[te].reset_index(drop=True)

        baseline_probs: dict[str, np.ndarray] = {}
        baseline_specs: list[tuple[str, str, np.ndarray, np.ndarray]] = [
            ("object_prior_allclass", "prior_object_id_object_part_allclass", np.empty((0, 0)), np.empty((0, 0))),
            ("object_prior_active_conditional", "prior_object_id_object_part_active", np.empty((0, 0)), np.empty((0, 0))),
            ("object_only_prior_allclass", "prior_object_id_allclass", np.empty((0, 0)), np.empty((0, 0))),
            ("object_only_prior_active_conditional", "prior_object_id_active", np.empty((0, 0)), np.empty((0, 0))),
            ("metadata_extra_trees", "extra_trees", meta_x[tr], meta_x[te]),
            ("metadata_logreg", "logreg", meta_x[tr], meta_x[te]),
            ("assist_full_extra_trees", "extra_trees", np.asarray(assist[tr], dtype=np.float32), np.asarray(assist[te], dtype=np.float32)),
            ("assist_tail_extra_trees", "extra_trees", assist_tail[tr], assist_tail[te]),
            ("assist_prefix_extra_trees", "extra_trees", assist_prefix[tr], assist_prefix[te]),
        ]
        for name, kind, x_tr, x_te in baseline_specs:
            if name in {"object_prior_allclass", "object_prior_active_conditional"}:
                prob = conditioned_prior_predict(
                    meta_tr,
                    y_tr,
                    meta_te,
                    cols=["object_id", "object_part"],
                    n_classes=n_classes,
                    active_only=name.endswith("active_conditional"),
                )
            elif name in {"object_only_prior_allclass", "object_only_prior_active_conditional"}:
                prob = conditioned_prior_predict(
                    meta_tr,
                    y_tr,
                    meta_te,
                    cols=["object_id"],
                    n_classes=n_classes,
                    active_only=name.endswith("active_conditional"),
                )
            else:
                prob = fit_predict_classifier(kind, x_tr, y_tr, x_te, n_classes=n_classes, seed=args.seed + fold.fold_id, n_jobs=args.n_jobs)
            baseline_probs[name] = prob
            met = prediction_metrics(y_te, prob)
            fold_rows.append({"audit": "context_prior_or_assist_ablation", "split": fold.split, "fold_id": fold.fold_id, "model": name, **met, "n_test": int(len(te))})
            pred = prob.argmax(axis=1)
            for subject, idx in meta_te.groupby("subject_id", sort=True).indices.items():
                ii = np.asarray(idx, dtype=int)
                subject_rows.append({"audit": "context_prior_or_assist_ablation", "split": fold.split, "fold_id": fold.fold_id, "subject_id": str(subject), "model": name, **prediction_metrics(y_te[ii], prob[ii]), "n": int(len(ii))})
            cm = confusion_matrix(y_te, pred, labels=np.arange(n_classes))
            for cls in range(n_classes):
                cls_mask = y_te == cls
                class_rows.append({
                    "audit": "context_prior_or_assist_ablation",
                    "split": fold.split,
                    "fold_id": fold.fold_id,
                    "model": name,
                    "class_id": int(cls),
                    "support": int(cls_mask.sum()),
                    "class_accuracy": float(np.mean(pred[cls_mask] == cls)) if cls_mask.any() else float("nan"),
                    "tp": int(cm[cls, cls]),
                    "fn": int(cm[cls, :].sum() - cm[cls, cls]),
                    "fp": int(cm[:, cls].sum() - cm[cls, cls]),
                })

        for user_model in USER_MODELS:
            p_user = fit_predict_classifier(user_model, np.asarray(user[tr], dtype=np.float32), y_tr, np.asarray(user[te], dtype=np.float32), n_classes=n_classes, seed=args.seed + fold.fold_id + 101, n_jobs=args.n_jobs)
            met = prediction_metrics(y_te, p_user)
            fold_rows.append({"audit": "user_decoder_adequacy", "split": fold.split, "fold_id": fold.fold_id, "model": user_model, **met, "n_test": int(len(te))})
            pred = p_user.argmax(axis=1)
            for subject, idx in meta_te.groupby("subject_id", sort=True).indices.items():
                ii = np.asarray(idx, dtype=int)
                subject_rows.append({"audit": "user_decoder_adequacy", "split": fold.split, "fold_id": fold.fold_id, "subject_id": str(subject), "model": user_model, **prediction_metrics(y_te[ii], p_user[ii]), "n": int(len(ii))})
            cm = confusion_matrix(y_te, pred, labels=np.arange(n_classes))
            for cls in range(n_classes):
                cls_mask = y_te == cls
                class_rows.append({
                    "audit": "user_decoder_adequacy",
                    "split": fold.split,
                    "fold_id": fold.fold_id,
                    "model": user_model,
                    "class_id": int(cls),
                    "support": int(cls_mask.sum()),
                    "class_accuracy": float(np.mean(pred[cls_mask] == cls)) if cls_mask.any() else float("nan"),
                    "tp": int(cm[cls, cls]),
                    "fn": int(cm[cls, :].sum() - cm[cls, cls]),
                    "fp": int(cm[:, cls].sum() - cm[cls, cls]),
                })

        # Policy-layer audits use logreg decoders because they provide calibrated-ish posterior traces
        # and are computationally stable across all outer folds.
        if len(dev) == 0:
            warnings.append({"split": fold.split, "fold_id": fold.fold_id, "warning": "empty dev split; using train split as calibration dev"})
            dev = tr
            y_dev = y_tr
            meta_dev = meta_tr
        pu_dev_raw = fit_predict_classifier("logreg", np.asarray(user[tr], dtype=np.float32), y_tr, np.asarray(user[dev], dtype=np.float32), n_classes=n_classes, seed=args.seed + fold.fold_id + 201, n_jobs=args.n_jobs)
        pu_test_raw = fit_predict_classifier("logreg", np.asarray(user[tr], dtype=np.float32), y_tr, np.asarray(user[te], dtype=np.float32), n_classes=n_classes, seed=args.seed + fold.fold_id + 202, n_jobs=args.n_jobs)
        pa_dev_raw = fit_predict_classifier("logreg", np.asarray(assist[tr], dtype=np.float32), y_tr, np.asarray(assist[dev], dtype=np.float32), n_classes=n_classes, seed=args.seed + fold.fold_id + 203, n_jobs=args.n_jobs)
        pa_test_raw = fit_predict_classifier("logreg", np.asarray(assist[tr], dtype=np.float32), y_tr, np.asarray(assist[te], dtype=np.float32), n_classes=n_classes, seed=args.seed + fold.fold_id + 204, n_jobs=args.n_jobs)

        calibration_pairs = {"none": (pu_test_raw, pa_test_raw)}
        pu_temp, t_u = calibrate_temperature(pu_dev_raw, y_dev, pu_test_raw)
        pa_temp, t_a = calibrate_temperature(pa_dev_raw, y_dev, pa_test_raw)
        calibration_pairs["scalar_temperature"] = (pu_temp, pa_temp)
        pu_dir, pu_dir_method = calibrate_dirichlet(pu_dev_raw, y_dev, pu_test_raw, args.seed + fold.fold_id)
        pa_dir, pa_dir_method = calibrate_dirichlet(pa_dev_raw, y_dev, pa_test_raw, args.seed + fold.fold_id + 1)
        calibration_pairs["dirichlet_logreg"] = (pu_dir, pa_dir)
        for cal_name, (pu_cal, pa_cal) in calibration_pairs.items():
            for branch_name, p in [("user", pu_cal), ("assist", pa_cal)]:
                calibration_rows.append({
                    "split": fold.split,
                    "fold_id": fold.fold_id,
                    "calibration": cal_name,
                    "branch": branch_name,
                    **prediction_metrics(y_te, p),
                    "temperature_user": t_u if cal_name == "scalar_temperature" else float("nan"),
                    "temperature_assist": t_a if cal_name == "scalar_temperature" else float("nan"),
                    "dirichlet_user_method": pu_dir_method if cal_name == "dirichlet_logreg" else "",
                    "dirichlet_assist_method": pa_dir_method if cal_name == "dirichlet_logreg" else "",
                })

        pu_test, pa_test = calibration_pairs["scalar_temperature"]
        pu_dev = apply_temperature(pu_dev_raw, t_u)
        pa_dev = apply_temperature(pa_dev_raw, t_a)

        deg_specs: list[tuple[str, dict[str, float | int]]] = [("clean", {})]
        deg_specs += [("uniform_mix", {"lam": lam}) for lam in MIX_LAMBDAS if lam > 0]
        deg_specs += [("trial_shift", {"shift": s}) for s in SHIFT_WINDOWS if s > 0]
        deg_specs += [("within_subject_shuffle", {})]
        deg_specs += [("global_shuffle", {})]

        g_u = pu_test.argmax(axis=1)
        user_score = pu_test.max(axis=1)
        policy_rows.append({"audit": "context_degradation", "split": fold.split, "fold_id": fold.fold_id, "condition": "clean", "policy": "UserOnly", **policy_metrics(y_te, pu_test, pa_test, g_u, np.zeros(len(g_u), dtype=bool), user_score), "n_test": int(len(te))})
        for condition, kwargs in deg_specs:
            pa_deg = degrade_prob(pa_test, meta_te, condition, seed=args.seed + 37 * fold.fold_id, n_classes=n_classes, **kwargs)
            g_a = pa_deg.argmax(axis=1)
            assist_score = pa_deg.max(axis=1)
            policy_rows.append({"audit": "context_degradation", "split": fold.split, "fold_id": fold.fold_id, "condition": condition_name(condition, kwargs), "policy": "AssistOnly", **policy_metrics(y_te, pu_test, pa_deg, g_a, g_u != g_a, assist_score), "n_test": int(len(te))})
            for tau in TAU_GRID:
                pred, take, score = cd_margin_policy(pu_test, pa_deg, tau)
                row = {"audit": "context_degradation", "split": fold.split, "fold_id": fold.fold_id, "condition": condition_name(condition, kwargs), "policy": f"CD-margin tau={tau:.2f}", "tau": tau, **policy_metrics(y_te, pu_test, pa_deg, pred, take, score), "n_test": int(len(te))}
                policy_rows.append(row)
                pareto_rows.append(row.copy())

        for budget_type, budget in [("cdc", 0.02), ("cdc", 0.05), ("intervention", 0.10), ("intervention", 0.25)]:
            thr = select_dev_threshold(y_dev, pu_dev, pa_dev, metric=budget_type, budget=budget)
            dev_pred, dev_take, dev_score = conf_gate_policy(pu_dev, pa_dev, thr)
            dev_pm = policy_metrics(y_dev, pu_dev, pa_dev, dev_pred, dev_take, dev_score)
            pred, take, score = conf_gate_policy(pu_test, pa_test, thr)
            test_pm = policy_metrics(y_te, pu_test, pa_test, pred, take, score)
            if budget_type == "cdc":
                dev_budget_value = dev_pm["expected_cdc"]
                test_budget_value = test_pm["expected_cdc"]
            else:
                dev_budget_value = dev_pm["intervention_rate"]
                test_budget_value = test_pm["intervention_rate"]
            policy_rows.append({
                "audit": "prospective_threshold",
                "split": fold.split,
                "fold_id": fold.fold_id,
                "condition": "clean",
                "policy": f"ConfGate dev {budget_type}<={budget:.2f}",
                "threshold": thr,
                "budget_type": budget_type,
                "budget": budget,
                "dev_budget_value": float(dev_budget_value),
                "test_budget_value": float(test_budget_value),
                "test_budget_gap": float(test_budget_value - budget),
                **test_pm,
                "n_test": int(len(te)),
            })
            pareto_rows.append(policy_rows[-1].copy())

    fold_df = pd.DataFrame(fold_rows)
    subject_df = pd.DataFrame(subject_rows)
    class_df = pd.DataFrame(class_rows)
    policy_df = pd.DataFrame(policy_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    pareto_df = pd.DataFrame(pareto_rows)

    fold_df.to_csv(out_dir / "db10_risk_boundary_fold_metrics.csv", index=False)
    subject_df.to_csv(out_dir / "db10_risk_boundary_subject_metrics.csv", index=False)
    class_df.to_csv(out_dir / "db10_risk_boundary_class_metrics.csv", index=False)
    policy_df.to_csv(out_dir / "db10_risk_boundary_policy_metrics.csv", index=False)
    calibration_df.to_csv(out_dir / "db10_risk_boundary_calibration_metrics.csv", index=False)
    pd.DataFrame(warnings, columns=["split", "fold_id", "warning"]).to_csv(out_dir / "db10_risk_boundary_warnings.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for keys, block in fold_df.groupby(["audit", "split", "model"], sort=True):
        audit, split, model = keys
        row = {"audit": audit, "split": split, "model": model, "n_folds": int(len(block))}
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "active_macro_f1", "nll", "ece", "brier"]:
            mean, lo, hi = mean_ci(block[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "db10_risk_boundary_summary.csv", index=False)

    policy_summary_rows: list[dict[str, Any]] = []
    for keys, block in policy_df.groupby(["audit", "split", "condition", "policy"], sort=True):
        audit, split, condition, policy = keys
        row = {"audit": audit, "split": split, "condition": condition, "policy": policy, "n_folds": int(len(block))}
        for metric in ["accuracy", "active_macro_f1", "risk_coverage_auc", "mean_cdc", "expected_cdc", "mean_cdc_all", "candidate_cdc_all", "intervention_rate"]:
            mean, lo, hi = mean_ci(block[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        policy_summary_rows.append(row)
    policy_summary = pd.DataFrame(policy_summary_rows)
    policy_summary.to_csv(out_dir / "db10_risk_boundary_policy_summary.csv", index=False)

    nd_rows: list[pd.DataFrame] = []
    if not pareto_df.empty:
        for keys, block in pareto_df[pareto_df["condition"] == "clean"].groupby(["split", "fold_id"], sort=True):
            nd = nondominated(block)
            nd["pareto"] = True
            nd_rows.append(nd)
        pareto_out = pd.concat(nd_rows, ignore_index=True) if nd_rows else pd.DataFrame()
        pareto_out.to_csv(out_dir / "db10_risk_boundary_pareto_nondominated_points.csv", index=False)
        pareto_summary = (
            pareto_out.groupby(["split", "policy"], as_index=False)
            .size()
            .rename(columns={"size": "n_nondominated_fold_points"})
            if not pareto_out.empty
            else pd.DataFrame(columns=["split", "policy", "n_nondominated_fold_points"])
        )
        pareto_summary.to_csv(out_dir / "db10_risk_boundary_pareto_summary.csv", index=False)

    make_figures(summary_df, policy_summary, calibration_df, out_dir)
    write_latex_tables(summary_df, policy_summary, calibration_df, out_dir)

    manifest = {
        "timestamp_utc": utc_now(),
        "processed_root": str(processed_root),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "n_rows": int(len(meta)),
        "n_subjects": int(meta["subject_id"].nunique()),
        "user_feature_dim": int(user.shape[1]),
        "assist_feature_dim": int(assist.shape[1]),
        "assist_tail_dim": int(assist.shape[1] - user.shape[1]),
        "metadata_cols_used": meta_cols,
        "leakage_excluded_metadata": sorted(LEAKAGE_EXCLUDED_METADATA),
        "splits": splits,
        "n_folds_run": int(len(folds)),
        "tau_grid": TAU_GRID,
        "mix_lambdas": MIX_LAMBDAS,
        "shift_windows": SHIFT_WINDOWS,
        "outputs": sorted([p.name for p in out_dir.glob("*") if p.is_file()]),
    }
    write_json(out_dir / "db10_risk_boundary_robustness_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def condition_name(condition: str, kwargs: dict[str, float | int]) -> str:
    if condition == "uniform_mix":
        return f"uniform_mix_lambda={float(kwargs.get('lam', 0.0)):.2f}"
    if condition == "trial_shift":
        return f"trial_shift_k={int(kwargs.get('shift', 0))}"
    return condition


def make_figures(summary: pd.DataFrame, policy_summary: pd.DataFrame, calibration: pd.DataFrame, out_dir: Path) -> None:
    if not summary.empty:
        focus = summary[
            (summary["audit"].isin(["context_prior_or_assist_ablation", "user_decoder_adequacy"]))
            & (summary["split"].isin(["amputee_loso", "able_to_amputee", "all_subject_kfold"]))
        ].copy()
        if not focus.empty:
            focus["label"] = focus["split"] + "\n" + focus["model"]
            fig, ax = plt.subplots(figsize=(11, 5.2))
            x = np.arange(len(focus))
            ax.bar(x, focus["active_macro_f1_mean"], color="#336699", alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(focus["label"], rotation=55, ha="right", fontsize=7)
            ax.set_ylabel("Held-out active macro-F1")
            ax.set_title("DB10 context-prior and user-decoder adequacy audits")
            ax.grid(axis="y", linewidth=0.3, alpha=0.35)
            fig.tight_layout()
            fig.savefig(out_dir / "F_db10_context_user_diagnostics.png", dpi=1000, bbox_inches="tight")
            fig.savefig(out_dir / "F_db10_context_user_diagnostics.pdf", bbox_inches="tight")
            plt.close(fig)

    if not policy_summary.empty:
        block = policy_summary[
            (policy_summary["audit"] == "context_degradation")
            & (policy_summary["split"] == "able_to_amputee")
            & (policy_summary["policy"] == "CD-margin tau=0.10")
        ].copy()
        if block.empty:
            block = policy_summary[
                (policy_summary["audit"] == "context_degradation")
                & (policy_summary["policy"] == "CD-margin tau=0.10")
            ].copy()
        if not block.empty:
            fig, ax = plt.subplots(figsize=(8.8, 4.8))
            block = block.sort_values("condition")
            x = np.arange(len(block))
            ax.plot(x, block["active_macro_f1_mean"], marker="o", color="#aa6633", label="Active F1")
            ax.set_xticks(x)
            ax.set_xticklabels(block["condition"], rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Active macro-F1")
            ax.set_title("DB10 context-degradation sensitivity, CD-margin tau=0.10")
            ax.grid(linewidth=0.3, alpha=0.35)
            fig.tight_layout()
            fig.savefig(out_dir / "F_db10_context_degradation.png", dpi=1000, bbox_inches="tight")
            fig.savefig(out_dir / "F_db10_context_degradation.pdf", bbox_inches="tight")
            plt.close(fig)

    if not calibration.empty:
        cal = (
            calibration.groupby(["split", "calibration", "branch"], as_index=False)
            .agg(ece_mean=("ece", "mean"), nll_mean=("nll", "mean"), active_macro_f1_mean=("active_macro_f1", "mean"))
        )
        fig, ax = plt.subplots(figsize=(9, 4.8))
        assist = cal[cal["branch"] == "assist"].copy()
        if not assist.empty:
            assist["label"] = assist["split"] + "\n" + assist["calibration"]
            x = np.arange(len(assist))
            ax.bar(x, assist["ece_mean"], color="#447744", alpha=0.88)
            ax.set_xticks(x)
            ax.set_xticklabels(assist["label"], rotation=55, ha="right", fontsize=7)
            ax.set_ylabel("Expected calibration error")
            ax.set_title("DB10 assistive-branch calibration-family sensitivity")
            ax.grid(axis="y", linewidth=0.3, alpha=0.35)
            fig.tight_layout()
            fig.savefig(out_dir / "F_db10_calibration_sensitivity.png", dpi=1000, bbox_inches="tight")
            fig.savefig(out_dir / "F_db10_calibration_sensitivity.pdf", bbox_inches="tight")
        plt.close(fig)


def write_latex_tables(summary: pd.DataFrame, policy_summary: pd.DataFrame, calibration: pd.DataFrame, out_dir: Path) -> None:
    def fmt(x: float) -> str:
        return "--" if pd.isna(x) else f"{float(x):.4f}"

    def esc(x: Any) -> str:
        return str(x).replace("_", r"\_")

    rows = []
    if not summary.empty:
        focus_models = [
            "object_prior_allclass",
            "object_prior_active_conditional",
            "object_only_prior_allclass",
            "object_only_prior_active_conditional",
            "metadata_extra_trees",
            "metadata_logreg",
            "assist_full_extra_trees",
            "assist_tail_extra_trees",
            "assist_prefix_extra_trees",
            "lda",
            "logreg",
            "random_forest",
            "extra_trees",
        ]
        focus = summary[summary["model"].isin(focus_models)].copy()
        focus = focus.sort_values(["split", "audit", "model"])
        for _, r in focus.iterrows():
            rows.append(
                f"{esc(r['split'])} & {esc(r['audit'])} & {esc(r['model'])} & "
                f"{fmt(r['active_macro_f1_mean'])} & {fmt(r['ece_mean'])} & {fmt(r['nll_mean'])} \\\\"
            )
    tex = (
        "\\begin{table*}[!t]\n\\centering\n\\caption{DB10 risk-boundary audits for context priors, metadata-only nonlinear baselines, assistive-feature ablations, and user-decoder adequacy. Values are averaged over held-out subject folds.}\n"
        "\\label{tab:db10-reader-robustness}\n\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lllrrr}\n\\toprule\nSplit & Audit & Model & Active F1 & ECE & NLL \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    (out_dir / "T_db10_risk_boundary_robustness_table.tex").write_text(tex, encoding="utf-8")

    prow = []
    if not policy_summary.empty:
        focus = policy_summary[
            (policy_summary["audit"].isin(["context_degradation", "prospective_threshold"]))
            & (
                policy_summary["policy"].isin(["CD-margin tau=0.10", "CD-margin tau=0.20", "AssistOnly", "UserOnly"])
                | policy_summary["policy"].str.startswith("ConfGate dev", na=False)
            )
        ].copy()
        selected_conditions = {
            "clean",
            "uniform_mix_lambda=0.50",
            "uniform_mix_lambda=1.00",
            "trial_shift_k=10",
            "within_subject_shuffle",
            "global_shuffle",
        }
        focus = focus[(focus["audit"] == "prospective_threshold") | (focus["condition"].isin(selected_conditions))]
        focus = focus.sort_values(["split", "audit", "condition", "policy"])
        for _, r in focus.iterrows():
            prow.append(
                f"{esc(r['split'])} & {esc(r['condition'])} & {esc(r['policy'])} & "
                f"{fmt(r['active_macro_f1_mean'])} & {fmt(r['risk_coverage_auc_mean'])} & {fmt(r['expected_cdc_mean'])} & {fmt(r['intervention_rate_mean'])} \\\\"
            )
    tex2 = (
        "\\begin{table*}[!t]\n\\centering\n\\caption{Selected DB10 policy-layer robustness rows under assistive-posterior sensitivity checks and prospective development-subject confidence thresholds. Uniform mixing flattens posterior confidence; shuffling and shifting test row alignment. Lower risk-coverage area and lower expected CDC are better. Full rows are in the CSV outputs.}\n"
        "\\label{tab:db10-policy-robustness}\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lllrrrr}\n\\toprule\nSplit & Condition & Policy & Active F1 & Risk area & Expected CDC & Intervention \\\\\n\\midrule\n"
        + "\n".join(prow)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    (out_dir / "T_db10_policy_robustness_table.tex").write_text(tex2, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DB10 risk-boundary robustness audits for the command-departure benchmark.")
    ap.add_argument("--processed_root", type=Path, default=Path("data/processed"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/db10_risk_boundary_audit"))
    ap.add_argument("--splits", default="amputee_loso,able_to_amputee,all_subject_kfold")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=0, help="Optional smoke-test cap across all folds; 0 means all folds.")
    ap.add_argument("--seed", type=int, default=20260512)
    ap.add_argument("--n_jobs", type=int, default=8)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
