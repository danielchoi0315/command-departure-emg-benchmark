from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
import yaml

from command_departure_benchmark.eval.metrics import accuracy, ece
from command_departure_benchmark.eval.splits import SubjectKFold
from command_departure_benchmark.models.calibrators import TempScaler
from command_departure_benchmark.utils.autobatch import autobatch_size, make_autobatch_record


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _predict_logits_batched(
    model: nn.Module,
    X: np.ndarray,
    *,
    device: str,
    batch_size: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), max(1, int(batch_size))):
            stop = min(len(X), start + max(1, int(batch_size)))
            xb = torch.tensor(X[start:stop], dtype=torch.float32).to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
            parts.append(logits.detach().float().cpu().numpy())
    if not parts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(parts)


def extract_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "g_star" not in df.columns:
        raise ValueError("windows.parquet missing required column g_star")
    y = df["g_star"].astype(int).to_numpy()
    subjects = df.get("subject_id", pd.Series(["S0"] * len(df))).astype(str).to_numpy()

    if "X_pu" in df.columns:
        X = np.vstack(df["X_pu"].apply(lambda v: np.asarray(v, dtype=float).reshape(-1)))
    elif "emg_features" in df.columns:
        # Backward-compatible fallback for pre-e2e windows.
        X = np.vstack(df["emg_features"].apply(lambda v: np.asarray(v, dtype=float).reshape(-1)))
    else:
        emg_cols = [c for c in df.columns if c.lower().startswith("emg_feat_") or c.lower().startswith("emg_ch")]
        if emg_cols:
            X = df[emg_cols].astype(float).to_numpy()
        else:
            raise ValueError("No p_u feature columns found (expected X_pu or emg_features/emg_feat_*/emg_ch*).")
    return X.astype(np.float32), y.astype(int), subjects


def split_train_val(train_idx: np.ndarray, subjects: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_subs = np.unique(subjects[train_idx])
    rng.shuffle(train_subs)
    n_val = max(1, int(round(0.2 * len(train_subs)))) if len(train_subs) > 1 else 1
    val_subs = set(train_subs[:n_val].tolist())
    val_mask = np.isin(subjects[train_idx], list(val_subs))
    val_idx = train_idx[val_mask]
    tr_idx = train_idx[~val_mask]
    if len(tr_idx) == 0:
        tr_idx = val_idx
    if len(val_idx) == 0:
        val_idx = tr_idx[: max(1, len(tr_idx) // 5)]
    return tr_idx, val_idx


def subsample_indices(indices: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or len(indices) <= max_rows:
        return indices
    rng = np.random.default_rng(seed)
    sampled = rng.choice(indices, size=max_rows, replace=False)
    return np.sort(sampled)


def nll_and_brier(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> tuple[float, float]:
    y_onehot = np.eye(n_classes)[y_true]
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
    nll = float(log_loss(y_true, probs, labels=list(range(n_classes))))
    return nll, brier


def reliability_curve(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": 0, "accuracy": 0.0, "confidence": 0.0})
            continue
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": int(mask.sum()),
                "accuracy": float(correct[mask].mean()),
                "confidence": float(conf[mask].mean()),
            }
        )
    return rows


def _as_logits_from_proba(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(proba, 1e-12, 1.0))


def _align_proba_to_global_classes(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    aligned = np.full((proba.shape[0], n_classes), 1e-12, dtype=float)
    cls = np.asarray(classes, dtype=int)
    valid = (cls >= 0) & (cls < n_classes)
    cls = cls[valid]
    if cls.size > 0:
        aligned[:, cls] = proba[:, valid]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


class TinyTCN(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, width: int = 128, depth: int = 8, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(1, width, kernel_size=3, padding=1), nn.ReLU()]
        for _ in range(max(1, depth - 1)):
            layers.extend(
                [
                    nn.Conv1d(width, width, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_classes)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        z = self.net(x).squeeze(-1)
        return self.head(z)


class TinyTransformer(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, width: int = 128, depth: int = 4, dropout: float = 0.2):
        super().__init__()
        self.n_tokens = min(16, max(1, in_dim))
        self.token_dim = int(np.ceil(in_dim / self.n_tokens))
        self.input_dim = self.n_tokens * self.token_dim

        self.proj = nn.Linear(self.token_dim, width)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, width))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=max(1, min(8, width // 32)) if width >= 32 else 1,
            dropout=dropout,
            dim_feedforward=width * 4,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, depth))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d = x.shape
        if d < self.input_dim:
            pad = torch.zeros((b, self.input_dim - d), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif d > self.input_dim:
            x = x[:, : self.input_dim]
        x = x.view(b, self.n_tokens, self.token_dim)
        z = self.proj(x) + self.pos
        z = self.encoder(z)
        z = self.norm(z.mean(dim=1))
        return self.head(z)


def build_deep_model(model_name: str, in_dim: int, n_classes: int, *, width: int, depth: int, dropout: float) -> nn.Module:
    if model_name == "tcn_deep":
        return TinyTCN(in_dim=in_dim, n_classes=n_classes, width=width, depth=depth, dropout=dropout)
    if model_name == "transformer_small":
        return TinyTransformer(in_dim=in_dim, n_classes=n_classes, width=width, depth=depth, dropout=dropout)
    raise ValueError(f"Unknown deep model: {model_name}")


def _set_torch_determinism(mode: str, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if mode == "correctness":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True


def train_deep_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    *,
    n_classes: int,
    mode: str,
    seed: int,
    dataset_name: str,
    fold_id: int,
    lr: float,
    width: int,
    depth: int,
    dropout: float,
    epochs: int,
    batch_size_cfg: int | str,
    precision: str,
    compile_enabled: bool,
    compile_mode: str,
    dataloader_workers: int,
    dataloader_pin_memory: bool,
    dataloader_persistent_workers: bool,
    dataloader_prefetch_factor: int,
    target_vram_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _set_torch_determinism(mode=mode, seed=seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compile_requested = bool(compile_enabled)
    compile_applied = False

    sample_rows = min(64, len(X_train))
    sample_batch = torch.tensor(X_train[:sample_rows], dtype=torch.float32)
    warmup_model = build_deep_model(
        model_name,
        in_dim=X_train.shape[1],
        n_classes=n_classes,
        width=width,
        depth=depth,
        dropout=dropout,
    ).to(device)
    auto = autobatch_size(
        warmup_model,
        sample_batch,
        mode=mode,
        requested_batch_size=batch_size_cfg,
        target_utilization=target_vram_fraction,
        correctness_batch=64,
        cpu_default_batch=256,
    )
    batch_size = int(auto.batch_size)
    del warmup_model
    if device == "cuda":
        torch.cuda.empty_cache()

    precision = precision.lower().strip()
    use_amp = False
    amp_dtype = torch.float32
    if device == "cuda" and mode == "throughput":
        if precision in {"bf16", "bfloat16"}:
            use_amp = True
            amp_dtype = torch.bfloat16
        elif precision in {"fp16", "float16"}:
            use_amp = True
            amp_dtype = torch.float16

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    num_workers = 0 if mode == "correctness" else max(0, int(dataloader_workers))
    pin_memory = bool(dataloader_pin_memory) and device == "cuda"
    prefetch_factor = max(2, int(dataloader_prefetch_factor)) if num_workers > 0 else None
    compile_current = compile_requested
    batch_size_current = max(1, int(batch_size))
    workers_current = int(num_workers)
    prefetch_current = prefetch_factor
    oom_retries = 0

    while True:
        data_gen = torch.Generator()
        data_gen.manual_seed(seed)
        model = build_deep_model(
            model_name,
            in_dim=X_train.shape[1],
            n_classes=n_classes,
            width=width,
            depth=depth,
            dropout=dropout,
        ).to(device)
        compile_applied = False
        if compile_current and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode=compile_mode)
                compile_applied = True
            except Exception as exc:
                print(f"[TRAIN_PU] {dataset_name}: fold={fold_id} compile disabled ({exc})")
                compile_current = False
                compile_applied = False

        persistent_workers = bool(dataloader_persistent_workers) and workers_current > 0
        loader = DataLoader(
            dataset,
            batch_size=batch_size_current,
            shuffle=True,
            generator=data_gen,
            num_workers=workers_current,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_current if workers_current > 0 else None,
            drop_last=False,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            train_samples = 0
            train_steps = 0
            train_start = time.perf_counter()
            for _ in range(epochs):
                model.train()
                for xb_cpu, yb_cpu in loader:
                    xb = xb_cpu.to(device, non_blocking=True)
                    yb = yb_cpu.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                        logits = model(xb)
                        loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()
                    train_samples += int(xb_cpu.shape[0])
                    train_steps += 1

            if device == "cuda":
                torch.cuda.synchronize()
            train_elapsed_s = max(1e-9, time.perf_counter() - train_start)
            train_samples_per_s = float(train_samples) / train_elapsed_s
            gpu_peak_mem_bytes = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0

            # Batched inference avoids giant one-shot GPU allocations on large folds.
            logits_val = _predict_logits_batched(
                model,
                X_val,
                device=device,
                batch_size=batch_size_current,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            logits_test = _predict_logits_batched(
                model,
                X_test,
                device=device,
                batch_size=batch_size_current,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            break
        except RuntimeError as exc:
            if device != "cuda" or not _is_cuda_oom(exc):
                raise
            oom_retries += 1
            changed = False
            if compile_current:
                compile_current = False
                changed = True
                reason = "disable_compile"
            elif batch_size_current > 128:
                batch_size_current = max(128, batch_size_current // 2)
                changed = True
                reason = f"batch={batch_size_current}"
            elif workers_current > 0:
                workers_current = max(0, workers_current // 2)
                prefetch_current = max(2, int(prefetch_current // 2)) if workers_current > 0 and prefetch_current else None
                changed = True
                reason = f"workers={workers_current}"
            else:
                reason = "no_further_backoff"
            print(
                f"[TRAIN_PU] {dataset_name}: fold={fold_id} {model_name} OOM retry={oom_retries} "
                f"next={reason} ({exc})"
            )
            if device == "cuda":
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            if not changed:
                raise
        finally:
            del loader
            del opt
            del loss_fn
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    if device == "cuda":
        torch.cuda.synchronize()

    auto_record = make_autobatch_record(auto, model_name=model_name, fold=fold_id, dataset=dataset_name)
    auto_record.update(
        {
            "epochs": int(epochs),
            "lr": float(lr),
            "width": int(width),
            "depth": int(depth),
            "dropout": float(dropout),
            "requested_batch_size": batch_size_cfg,
            "precision": precision,
            "amp_enabled": bool(use_amp),
            "compile_enabled": bool(compile_requested),
            "compile_applied": bool(compile_applied),
            "compile_mode": compile_mode,
            "dataloader_num_workers": int(workers_current),
            "dataloader_pin_memory": bool(pin_memory),
            "dataloader_persistent_workers": bool(dataloader_persistent_workers and workers_current > 0),
            "dataloader_prefetch_factor": int(prefetch_current) if prefetch_current is not None else 0,
            "train_steps": int(train_steps),
            "train_samples": int(train_samples),
            "train_elapsed_s": float(train_elapsed_s),
            "train_samples_per_s": float(train_samples_per_s),
            "gpu_peak_mem_bytes": int(gpu_peak_mem_bytes),
            "gpu_peak_mem_gb": float(gpu_peak_mem_bytes / (1024 ** 3)) if gpu_peak_mem_bytes else 0.0,
            "oom_retries": int(oom_retries),
            "final_batch_size": int(batch_size_current),
        }
    )
    peak_mem_gb = float(gpu_peak_mem_bytes / (1024 ** 3)) if gpu_peak_mem_bytes else 0.0
    print(
        "[TRAIN_PU][perf] "
        f"{dataset_name} fold={fold_id} model={model_name} "
        f"samples_per_s={train_samples_per_s:.2f} peak_mem_gb={peak_mem_gb:.3f} "
        f"workers={workers_current} batch={batch_size_current} amp={use_amp} "
        f"compile={compile_applied} oom_retries={oom_retries}"
    )
    return logits_val, logits_test, auto_record


def evaluate_model(
    model_name: str,
    *,
    logits_val: np.ndarray,
    y_val: np.ndarray,
    logits_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    ts = TempScaler()
    ts.fit(logits_val, y_val, max_iter=100)
    probs_before = np.exp(logits_test - logits_test.max(axis=1, keepdims=True))
    probs_before /= probs_before.sum(axis=1, keepdims=True)
    probs_after = ts.transform(logits_test)

    y_pred = probs_after.argmax(axis=1)
    nll, brier = nll_and_brier(probs_after, y_test, n_classes=n_classes)
    metrics = {
        "model": model_name,
        "accuracy": accuracy(y_test, y_pred),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "ece": ece(probs_after, y_test),
        "nll": nll,
        "brier": brier,
    }
    calib = {
        "model": model_name,
        "temperature": float(ts.temperature),
        "ece_before": ece(probs_before, y_test),
        "ece_after": metrics["ece"],
        "reliability_curve": reliability_curve(probs_after, y_test, n_bins=10),
    }
    return metrics, calib, probs_after


def _parse_model_list(exp_cfg: dict[str, Any], mode: str) -> list[str]:
    env_models = os.environ.get("PU_MODELS", "").strip()
    if env_models:
        out = [m.strip() for m in env_models.split(",") if m.strip()]
        if not out:
            out = ["lda", "logreg", "tcn_deep"]
    else:
        models = exp_cfg.get("pu_models")
        if isinstance(models, list) and models:
            out = [str(m) for m in models]
        else:
            out = ["lda", "logreg", "tcn_deep"]
    if mode == "correctness" and "transformer_small" in out and os.environ.get("RUN_DEEP_IN_CORRECTNESS", "0") != "1":
        # Keep correctness mode conservative by default.
        out = [m for m in out if m != "transformer_small"]
    return out


def _deep_cfg(exp_cfg: dict[str, Any]) -> dict[str, Any]:
    d = dict(exp_cfg.get("pu_deep", {}) or {})
    dl = dict(d.get("dataloader", exp_cfg.get("dataloader", {})) or {})
    batch_size_cfg = os.environ.get("PU_BATCH_SIZE", d.get("batch_size", exp_cfg.get("batch_size", "auto_max_vram")))
    precision_cfg = os.environ.get("PU_PRECISION", str(d.get("precision", exp_cfg.get("precision", "fp32")))).lower()
    compile_cfg = _env_bool("PU_COMPILE", bool(d.get("compile", exp_cfg.get("compile", False))))
    compile_mode_cfg = os.environ.get(
        "PU_COMPILE_MODE", str(d.get("compile_mode", exp_cfg.get("compile_mode", "max-autotune")))
    )
    max_train_rows_cfg = int(
        os.environ.get("PU_DEEP_MAX_TRAIN_ROWS", str(d.get("max_train_rows", os.environ.get("PU_DEEP_MAX_TRAIN_ROWS", "60000"))))
    )
    workers_cfg = int(
        os.environ.get("PU_DATALOADER_WORKERS", str(dl.get("workers", d.get("workers", max(1, min(16, os.cpu_count() or 4))))))
    )
    pin_cfg = _env_bool("PU_DATALOADER_PIN_MEMORY", bool(dl.get("pin_memory", d.get("pin_memory", True))))
    persistent_cfg = _env_bool(
        "PU_DATALOADER_PERSISTENT_WORKERS", bool(dl.get("persistent_workers", d.get("persistent_workers", True)))
    )
    prefetch_cfg = int(os.environ.get("PU_DATALOADER_PREFETCH_FACTOR", str(dl.get("prefetch_factor", d.get("prefetch_factor", 4)))))
    target_vram_cfg = float(os.environ.get("PU_TARGET_VRAM_FRACTION", str(d.get("target_vram_fraction", 0.85))))
    return {
        "lr": float(d.get("lr", 3e-4)),
        "dropout": float(d.get("dropout", 0.3)),
        "width": int(d.get("width", 128)),
        "depth": int(d.get("depth", 8)),
        "epochs": int(d.get("epochs", 10)),
        "batch_size": batch_size_cfg,
        "precision": precision_cfg,
        "compile": compile_cfg,
        "compile_mode": compile_mode_cfg,
        "max_train_rows": max_train_rows_cfg,
        "dataloader_workers": workers_cfg,
        "dataloader_pin_memory": pin_cfg,
        "dataloader_persistent_workers": persistent_cfg,
        "dataloader_prefetch_factor": prefetch_cfg,
        "target_vram_fraction": target_vram_cfg,
    }


def run_dataset(
    windows_path: Path,
    out_dir: Path,
    *,
    dataset_name: str,
    dataset_role: str,
    seed: int,
    mode: str,
    max_folds: int,
    exp_cfg: dict[str, Any],
) -> None:
    role = str(dataset_role).strip().lower() or "full"
    if role in {"p_a_only", "workload_only"} and os.environ.get("ALLOW_PU_ROLE_OVERRIDE", "0") != "1":
        print(f"[TRAIN_PU] {dataset_name}: skipped (dataset_role={role})")
        return

    df = pd.read_parquet(windows_path)
    X, y, subjects = extract_features(df)
    if len(np.unique(y)) < 2:
        return

    max_train_rows = int(
        os.environ.get("PU_MAX_TRAIN_ROWS", "150000" if mode == "throughput" else "100000")
    )

    models_requested = _parse_model_list(exp_cfg, mode=mode)
    deep_cfg = _deep_cfg(exp_cfg)
    deep_max_rows = int(deep_cfg.get("max_train_rows", 0))
    pred_model = str(os.environ.get("PU_PRED_MODEL", str(exp_cfg.get("pu_pred_model", "logreg"))))
    if pred_model not in models_requested:
        models_requested.append(pred_model)
    id_cols = [c for c in ["dataset", "subject_id", "session_id", "trial_id", "window_id", "g_star"] if c in df.columns]
    pu_pred_parts: list[pd.DataFrame] = []

    cv = SubjectKFold(n_splits=min(max_folds, max(2, len(np.unique(subjects)))), seed=seed)
    fold_metrics: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    autobatch_records: list[dict[str, Any]] = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(subjects.tolist())):
        tr_idx, val_idx = split_train_val(train_idx, subjects, seed + fold_id)
        tr_idx = subsample_indices(tr_idx, max_train_rows, seed + 17 * (fold_id + 1))
        val_idx = subsample_indices(val_idx, max_train_rows // 2, seed + 19 * (fold_id + 1))
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]
        Xte, yte = X[test_idx], y[test_idx]
        n_classes = int(y.max()) + 1

        if "lda" in models_requested:
            lda = LinearDiscriminantAnalysis()
            lda.fit(Xtr, ytr)
            lda_val_proba = _align_proba_to_global_classes(lda.predict_proba(Xval), lda.classes_, n_classes)
            lda_test_proba = _align_proba_to_global_classes(lda.predict_proba(Xte), lda.classes_, n_classes)
            lda_val = _as_logits_from_proba(lda_val_proba)
            lda_test = _as_logits_from_proba(lda_test_proba)
            lda_metrics, lda_cal, lda_probs = evaluate_model(
                "lda", logits_val=lda_val, y_val=yval, logits_test=lda_test, y_test=yte, n_classes=n_classes
            )
            lda_metrics["fold"] = fold_id
            lda_cal["fold"] = fold_id
            fold_metrics.append(lda_metrics)
            calibration.append(lda_cal)
            if pred_model == "lda":
                pred_df = df.iloc[test_idx][id_cols].reset_index(drop=True).copy()
                pred_df["fold_id"] = int(fold_id)
                pred_df["model"] = "lda"
                pred_df["p_u_pred"] = [row.astype(float).tolist() for row in lda_probs]
                pu_pred_parts.append(pred_df)

        if "logreg" in models_requested:
            lr = LogisticRegression(max_iter=2000)
            lr.fit(Xtr, ytr)
            lr_val_proba = _align_proba_to_global_classes(lr.predict_proba(Xval), lr.classes_, n_classes)
            lr_test_proba = _align_proba_to_global_classes(lr.predict_proba(Xte), lr.classes_, n_classes)
            lr_val = _as_logits_from_proba(lr_val_proba)
            lr_test = _as_logits_from_proba(lr_test_proba)
            lr_metrics, lr_cal, lr_probs = evaluate_model(
                "logreg", logits_val=lr_val, y_val=yval, logits_test=lr_test, y_test=yte, n_classes=n_classes
            )
            lr_metrics["fold"] = fold_id
            lr_cal["fold"] = fold_id
            fold_metrics.append(lr_metrics)
            calibration.append(lr_cal)
            if pred_model == "logreg":
                pred_df = df.iloc[test_idx][id_cols].reset_index(drop=True).copy()
                pred_df["fold_id"] = int(fold_id)
                pred_df["model"] = "logreg"
                pred_df["p_u_pred"] = [row.astype(float).tolist() for row in lr_probs]
                pu_pred_parts.append(pred_df)

        for deep_model in [m for m in models_requested if m in {"tcn_deep", "transformer_small"}]:
            run_deep = mode == "throughput" or os.environ.get("RUN_DEEP_IN_CORRECTNESS", "0") == "1"
            if not run_deep:
                continue
            Xtr_deep, ytr_deep = Xtr, ytr
            if deep_max_rows > 0 and len(Xtr_deep) > deep_max_rows:
                keep = subsample_indices(np.arange(len(Xtr_deep)), deep_max_rows, seed + 101 * (fold_id + 1))
                Xtr_deep = Xtr_deep[keep]
                ytr_deep = ytr_deep[keep]
                print(
                    f"[TRAIN_PU] {dataset_name}: fold={fold_id} {deep_model} "
                    f"subsampled train_rows={len(Xtr)} -> {len(Xtr_deep)}"
                )

            logits_val, logits_test, auto_record = train_deep_model(
                deep_model,
                Xtr_deep,
                ytr_deep,
                Xval,
                Xte,
                n_classes=n_classes,
                mode=mode,
                seed=seed + fold_id,
                dataset_name=dataset_name,
                fold_id=fold_id,
                lr=deep_cfg["lr"],
                width=deep_cfg["width"],
                depth=deep_cfg["depth"],
                dropout=deep_cfg["dropout"],
                epochs=deep_cfg["epochs"],
                batch_size_cfg=deep_cfg["batch_size"],
                precision=deep_cfg["precision"],
                compile_enabled=deep_cfg["compile"],
                compile_mode=deep_cfg["compile_mode"],
                dataloader_workers=deep_cfg["dataloader_workers"],
                dataloader_pin_memory=deep_cfg["dataloader_pin_memory"],
                dataloader_persistent_workers=deep_cfg["dataloader_persistent_workers"],
                dataloader_prefetch_factor=deep_cfg["dataloader_prefetch_factor"],
                target_vram_fraction=deep_cfg["target_vram_fraction"],
            )
            deep_metrics, deep_cal, deep_probs = evaluate_model(
                deep_model,
                logits_val=logits_val,
                y_val=yval,
                logits_test=logits_test,
                y_test=yte,
                n_classes=n_classes,
            )
            deep_metrics["fold"] = fold_id
            deep_cal["fold"] = fold_id
            fold_metrics.append(deep_metrics)
            calibration.append(deep_cal)
            autobatch_records.append(auto_record)
            if pred_model == deep_model:
                pred_df = df.iloc[test_idx][id_cols].reset_index(drop=True).copy()
                pred_df["fold_id"] = int(fold_id)
                pred_df["model"] = deep_model
                pred_df["p_u_pred"] = [row.astype(float).tolist() for row in deep_probs]
                pu_pred_parts.append(pred_df)

    if not fold_metrics:
        return
    if not pu_pred_parts:
        raise ValueError(
            f"No p_u predictions were produced for dataset={dataset_name}. "
            f"Requested pred model '{pred_model}' may be unavailable."
        )

    pu_pred = pd.concat(pu_pred_parts, ignore_index=True)
    if id_cols:
        if pu_pred.duplicated(subset=id_cols).any():
            dup_n = int(pu_pred.duplicated(subset=id_cols).sum())
            raise ValueError(f"{dataset_name}: pu_pred contains duplicate id rows (n={dup_n})")
        if len(pu_pred) != len(df):
            raise ValueError(
                f"{dataset_name}: pu_pred coverage mismatch rows={len(pu_pred)} expected={len(df)}"
            )

    fold_df = pd.DataFrame(fold_metrics)
    model_rows = []
    for model, block in fold_df.groupby("model"):
        model_rows.append(
            {
                "model": model,
                "accuracy_mean": float(block["accuracy"].mean()),
                "accuracy_std": float(block["accuracy"].std(ddof=1)) if len(block) > 1 else 0.0,
                "macro_f1_mean": float(block["macro_f1"].mean()),
                "macro_f1_std": float(block["macro_f1"].std(ddof=1)) if len(block) > 1 else 0.0,
                "ece_mean": float(block["ece"].mean()),
                "ece_std": float(block["ece"].std(ddof=1)) if len(block) > 1 else 0.0,
                "nll_mean": float(block["nll"].mean()),
                "nll_std": float(block["nll"].std(ddof=1)) if len(block) > 1 else 0.0,
                "brier_mean": float(block["brier"].mean()),
                "brier_std": float(block["brier"].std(ddof=1)) if len(block) > 1 else 0.0,
            }
        )
    metrics = {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
        "mode": mode,
        "seed": int(seed),
        "pu_models_requested": models_requested,
        "pu_pred_model": pred_model,
        "pu_deep_config": deep_cfg,
        "models": model_rows,
        "autobatch": autobatch_records,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "PRED"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pu_pred.to_parquet(pred_dir / "pu_pred.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2))
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    (out_dir / "train_pu_manifest.json").write_text(
        json.dumps(
            {
                "stage": "train_pu_dataset",
                "dataset": dataset_name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seed": int(seed),
                "mode": mode,
                "features_source": str(windows_path),
                "models": models_requested,
                "pu_pred_model": pred_model,
                "pu_pred_path": str(pred_dir / "pu_pred.parquet"),
                "autobatch": autobatch_records,
            },
            indent=2,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--datasets_yaml", type=Path, required=True)
    ap.add_argument("--exp_yaml", type=Path, required=True)
    ap.add_argument("--mode", choices=["correctness", "throughput"], default="throughput")
    ap.add_argument("--max_folds", type=int, default=int(os.environ.get("TRAIN_PU_MAX_FOLDS", "3")))
    args = ap.parse_args()

    dsreg = load_yaml(args.datasets_yaml)["datasets"]
    exp = load_yaml(args.exp_yaml)["experiment"]
    exp_id = os.environ.get("EXP_ID", exp["id"])

    data_root = Path(os.environ.get("DATA_ROOT", "lake"))
    results_root = Path("results") / exp_id
    derived_root = data_root / "derived"
    seed = int(exp.get("seed", 1337))

    for ds_id, info in dsreg.items():
        if int(info.get("tier", 99)) > args.tier:
            continue
        windows_path = derived_root / ds_id / info.get("version", "unknown") / "windows.parquet"
        if not windows_path.exists():
            continue
        out_dir = results_root / ds_id
        try:
            run_dataset(
                windows_path,
                out_dir,
                dataset_name=ds_id,
                dataset_role=str(info.get("arbitration_role", "full")),
                seed=seed,
                mode=args.mode,
                max_folds=args.max_folds,
                exp_cfg=exp,
            )
            print(f"[TRAIN_PU] {ds_id}: wrote metrics/calibration artifacts")
        except Exception as exc:
            print(f"[TRAIN_PU] {ds_id}: skipped ({exc})")


if __name__ == "__main__":
    main()
