from __future__ import annotations
import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error for multiclass probs using confidence=max prob."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc_bin = correct[m].mean()
        conf_bin = conf[m].mean()
        e += (m.mean()) * abs(acc_bin - conf_bin)
    return float(e)
