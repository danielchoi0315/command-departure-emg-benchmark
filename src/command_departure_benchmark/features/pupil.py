from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PupilPreprocSpec:
    fs: float
    blink_fill_max_s: float = 0.250
    zscore: bool = True

def simple_pupil_preproc(pupil: np.ndarray, spec: PupilPreprocSpec) -> np.ndarray:
    """Very simple preprocessing: interpolate short NaN gaps, optional z-score."""
    x = pupil.astype(float).copy()
    n = len(x)

    # Interpolate NaNs
    isn = np.isnan(x)
    if isn.any():
        idx = np.arange(n)
        good = ~isn
        if good.sum() >= 2:
            x[isn] = np.interp(idx[isn], idx[good], x[good])

    if spec.zscore:
        mu = np.mean(x)
        sd = np.std(x) + 1e-9
        x = (x - mu) / sd
    return x

def pupil_window_features(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "slope": float((x[-1] - x[0]) / max(1, len(x)-1)),
    }
