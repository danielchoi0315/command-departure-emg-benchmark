#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from command_departure_benchmark.eval.splits import SubjectKFold


VARIANT_ORDER = [
    "full_assist",
    "user_emg_only",
    "assist_prefix_only",
    "assist_tail_only",
    "metadata_object_context",
    "full_assist_shuffled_tail",
    "assist_tail_shuffled_only",
]

DISPLAY_LABELS = {
    "full_assist": "Full assist",
    "user_emg_only": "User EMG only",
    "assist_prefix_only": "Assist prefix only",
    "assist_tail_only": "Assist tail only",
    "metadata_object_context": "Metadata object context",
    "full_assist_shuffled_tail": "Full assist + shuffled tail",
    "assist_tail_shuffled_only": "Shuffled assist tail only",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _align_proba(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.full((proba.shape[0], n_classes), 1e-12, dtype=float)
    cls = np.asarray(classes, dtype=int)
    valid = (cls >= 0) & (cls < n_classes)
    if valid.any():
        out[:, cls[valid]] = proba[:, valid]
    out = np.clip(out, 1e-12, None)
    return out / out.sum(axis=1, keepdims=True)


def _prior(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(float)
    counts += 1e-3
    return counts / counts.sum()


def _metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1).astype(int)
    labels = np.arange(probs.shape[1])
    active = y != 0
    return {
        "accuracy": float(np.mean(pred == y)) if len(y) else float("nan"),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)) if len(y) else float("nan"),
        "active_macro_f1": float(f1_score(y[active], pred[active], average="macro", zero_division=0)) if active.any() else float("nan"),
        "nll": float(log_loss(y, probs, labels=labels)) if len(y) else float("nan"),
    }


def _stable_variant_offset(variant: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(variant)) % 997


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, *, n_classes: int) -> tuple[np.ndarray, str]:
    if train_x.shape[1] == 0 or len(np.unique(train_y)) < 2:
        p = _prior(train_y, n_classes)
        return np.repeat(p.reshape(1, -1), len(test_x), axis=0), "global_prior"
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=100, solver="newton-cholesky", tol=1e-4, class_weight=None),
    )
    clf.fit(train_x, train_y)
    model = clf[-1]
    probs = _align_proba(clf.predict_proba(test_x), model.classes_, n_classes)
    p = _prior(train_y, n_classes).reshape(1, -1)
    probs = 0.7 * probs + 0.3 * np.repeat(p, len(test_x), axis=0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, "standardized_logreg_smoothed"


def _metadata_object_context(meta: pd.DataFrame) -> np.ndarray:
    cols = [
        c
        for c in [
            "position_id",
            "object_id",
            "object_part",
            "dynamic_flag",
            "gaze_available",
            "pupil_left_available",
            "pupil_right_available",
            "acc_available",
            "gyr_available",
        ]
        if c in meta.columns
    ]
    if not cols:
        return np.zeros((len(meta), 0), dtype=float)
    block = meta[cols].copy()
    for c in block.columns:
        block[c] = block[c].astype(str).fillna("__missing__")
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return enc.fit_transform(block).astype(float)


def _subject_rows(
    *,
    meta: pd.DataFrame,
    y: np.ndarray,
    probs: np.ndarray,
    variant: str,
    fold_id: int,
    method: str,
) -> list[dict[str, Any]]:
    pred = probs.argmax(axis=1).astype(int)
    rows: list[dict[str, Any]] = []
    for subject, idx in meta.groupby("subject_id", sort=True).indices.items():
        ii = np.asarray(idx, dtype=int)
        m = _metrics(y[ii], probs[ii])
        rows.append(
            {
                "variant": variant,
                "fold_id": int(fold_id),
                "subject_id": str(subject),
                "method": method,
                "n_windows": int(len(ii)),
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "active_macro_f1": m["active_macro_f1"],
                "nll": m["nll"],
                "override_free_pred_rate": float(np.mean(pred[ii] != y[ii])),
            }
        )
    return rows


def _variant_features(
    *,
    variant: str,
    user: np.ndarray,
    assist_prefix: np.ndarray,
    assist_tail: np.ndarray,
    metadata_context: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if variant == "full_assist":
        return np.concatenate([assist_prefix, assist_tail], axis=1)
    if variant == "user_emg_only":
        return user
    if variant == "assist_prefix_only":
        return assist_prefix
    if variant == "assist_tail_only":
        return assist_tail
    if variant == "metadata_object_context":
        return metadata_context
    if variant == "full_assist_shuffled_tail":
        shuffled = assist_tail[rng.permutation(len(assist_tail))]
        return np.concatenate([assist_prefix, shuffled], axis=1)
    if variant == "assist_tail_shuffled_only":
        return assist_tail[rng.permutation(len(assist_tail))]
    raise ValueError(f"unknown variant: {variant}")


def _draw(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    block = summary.copy()
    block["variant"] = pd.Categorical(block["variant"], categories=VARIANT_ORDER, ordered=True)
    block = block.sort_values("variant")
    x = np.arange(len(block))
    ax.bar(x, block["active_macro_f1_mean"], color="#4477aa", alpha=0.88)
    yerr = np.vstack(
        [
            block["active_macro_f1_mean"] - block["active_macro_f1_ci_low"],
            block["active_macro_f1_ci_high"] - block["active_macro_f1_mean"],
        ]
    )
    ax.errorbar(x, block["active_macro_f1_mean"], yerr=yerr, fmt="none", ecolor="black", capsize=3, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS.get(str(v), str(v)).replace(" ", "\n") for v in block["variant"]], fontsize=7)
    ax.set_ylabel("Held-out active macro-F1")
    ax.set_title("DB10 assistive-branch context-control audit")
    ax.grid(axis="y", linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "F_db10_context_control_active_f1.png", dpi=1000, bbox_inches="tight")
    fig.savefig(out_dir / "F_db10_context_control_active_f1.pdf", bbox_inches="tight")
    plt.close(fig)


def _mean_ci(series: pd.Series) -> tuple[float, float, float]:
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, mean, mean
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _write_latex(summary: pd.DataFrame, out_dir: Path) -> None:
    block = summary.copy()
    block["variant"] = pd.Categorical(block["variant"], categories=VARIANT_ORDER, ordered=True)
    block = block.sort_values("variant")
    rows = []
    for _, r in block.iterrows():
        variant = DISPLAY_LABELS.get(str(r["variant"]), str(r["variant"])).replace("_", r"\_")
        rows.append(
            f"{variant} & {float(r['accuracy_mean']):.4f} & {float(r['active_macro_f1_mean']):.4f} & "
            f"{float(r['nll_mean']):.4f} & {int(r['n_subject_rows'])} \\\\"
        )
    tex = (
        r"""\begin{table*}[!t]
\centering
\caption{DB10 assistive-branch context-control audit. Models were fit on training subjects and evaluated on held-out subjects. Full assistive features use the processed 190-dimensional assistive branch; user-only uses the processed 96-dimensional user branch; prefix/tail controls split the assistive branch at the user-branch dimensionality. Shuffled controls preserve feature marginals while breaking row-level tail association.}
\label{tab:db10-context-control}
\scriptsize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrr}
\toprule
Feature variant & Accuracy & Active macro-F1 & NLL & Subject rows \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    )
    (out_dir / "T_db10_context_control_table.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="DB10 assistive-branch context-control audit.")
    ap.add_argument("--processed_root", type=Path, default=Path("data/processed"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/db10_context_control_audit"))
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260511)
    args = ap.parse_args()

    db10 = args.processed_root / "db10"
    meta = pd.read_csv(db10 / "metadata.csv")
    user = np.load(db10 / "user_features.npy")
    assist = np.load(db10 / "assist_features.npy")
    y = np.load(db10 / "labels.npy").astype(int)
    if len(meta) != len(user) or len(meta) != len(assist) or len(meta) != len(y):
        raise SystemExit("metadata/features/labels row counts do not match")
    if assist.shape[1] < user.shape[1]:
        raise SystemExit("assist_features must have at least as many columns as user_features for prefix/tail controls")

    assist_prefix = np.asarray(assist[:, : user.shape[1]], dtype=float)
    assist_tail = np.asarray(assist[:, user.shape[1] :], dtype=float)
    prefix_delta = np.asarray(assist_prefix - user, dtype=float)
    prefix_mean_abs_delta = float(np.nanmean(np.abs(prefix_delta)))
    prefix_max_abs_delta = float(np.nanmax(np.abs(prefix_delta)))
    metadata_context = _metadata_object_context(meta)
    n_classes = int(y.max()) + 1
    subjects = meta["subject_id"].astype(str).to_numpy()
    folds = list(
        SubjectKFold(n_splits=min(args.n_splits, meta["subject_id"].nunique()), seed=int(args.seed)).split(
            subjects.tolist()
        )
    )

    fold_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        train_y = y[train_idx]
        test_y = y[test_idx]
        for variant in VARIANT_ORDER:
            print(
                f"[db10-context] fold={fold_id + 1}/{len(folds)} variant={variant} "
                f"train={len(train_idx)} test={len(test_idx)}",
                flush=True,
            )
            offset = _stable_variant_offset(variant)
            rng_train = np.random.default_rng(int(args.seed) + fold_id * 1009 + offset)
            rng_test = np.random.default_rng(int(args.seed) + fold_id * 2003 + offset)
            x_train = _variant_features(
                variant=variant,
                user=np.asarray(user[train_idx], dtype=float),
                assist_prefix=np.asarray(assist_prefix[train_idx], dtype=float),
                assist_tail=np.asarray(assist_tail[train_idx], dtype=float),
                metadata_context=np.asarray(metadata_context[train_idx], dtype=float),
                rng=rng_train,
            )
            x_test = _variant_features(
                variant=variant,
                user=np.asarray(user[test_idx], dtype=float),
                assist_prefix=np.asarray(assist_prefix[test_idx], dtype=float),
                assist_tail=np.asarray(assist_tail[test_idx], dtype=float),
                metadata_context=np.asarray(metadata_context[test_idx], dtype=float),
                rng=rng_test,
            )
            probs, method = _fit_predict(x_train, train_y, x_test, n_classes=n_classes)
            m = _metrics(test_y, probs)
            print(
                f"[db10-context] done fold={fold_id + 1}/{len(folds)} variant={variant} "
                f"accuracy={m['accuracy']:.4f} active_macro_f1={m['active_macro_f1']:.4f}",
                flush=True,
            )
            fold_rows.append(
                {
                    "variant": variant,
                    "fold_id": int(fold_id),
                    "method": method,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_subjects": int(meta.iloc[train_idx]["subject_id"].nunique()),
                    "n_test_subjects": int(meta.iloc[test_idx]["subject_id"].nunique()),
                    **m,
                }
            )
            subject_rows.extend(
                _subject_rows(
                    meta=meta.iloc[test_idx].reset_index(drop=True),
                    y=test_y,
                    probs=probs,
                    variant=variant,
                    fold_id=fold_id,
                    method=method,
                )
            )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(fold_rows)
    subject_df = pd.DataFrame(subject_rows)
    fold_df.to_csv(out_dir / "context_control_fold_metrics.csv", index=False)
    subject_df.to_csv(out_dir / "context_control_subject_metrics.csv", index=False)

    rows = []
    for variant, block in subject_df.groupby("variant", sort=False):
        row: dict[str, Any] = {"variant": variant, "n_subject_rows": int(len(block))}
        for metric in ["accuracy", "macro_f1", "active_macro_f1", "nll"]:
            mean, lo, hi = _mean_ci(block[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["variant"] = pd.Categorical(summary["variant"], categories=VARIANT_ORDER, ordered=True)
    summary = summary.sort_values("variant").reset_index(drop=True)
    summary.to_csv(out_dir / "context_control_summary.csv", index=False)
    _draw(summary, out_dir)
    _write_latex(summary, out_dir)

    manifest = {
        "timestamp_utc": _utc_now(),
        "processed_root": str(args.processed_root),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "n_rows": int(len(meta)),
        "n_subjects": int(meta["subject_id"].nunique()),
        "user_feature_dim": int(user.shape[1]),
        "assist_feature_dim": int(assist.shape[1]),
        "assist_prefix_dim": int(assist_prefix.shape[1]),
        "assist_tail_dim": int(assist_tail.shape[1]),
        "metadata_context_dim": int(metadata_context.shape[1]),
        "assist_prefix_vs_user_mean_abs_delta": prefix_mean_abs_delta,
        "assist_prefix_vs_user_max_abs_delta": prefix_max_abs_delta,
        "feature_interpretation": "The processed assistive branch is audited directly. Its first user_dim columns are treated as an assistive-prefix control, not assumed to be identical to user_features.",
        "variants": VARIANT_ORDER,
    }
    _write_json(out_dir / "context_control_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
