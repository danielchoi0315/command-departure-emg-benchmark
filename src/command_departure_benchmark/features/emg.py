from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, filtfilt

@dataclass(frozen=True)
class EMGPreprocSpec:
    fs: float
    band_lo: float = 20.0
    band_hi: float = 450.0
    order: int = 4

def bandpass_rectify(x: np.ndarray, spec: EMGPreprocSpec) -> np.ndarray:
    """Bandpass + rectify. x shape: (T, C)"""
    nyq = 0.5 * spec.fs
    b, a = butter(spec.order, [spec.band_lo/nyq, spec.band_hi/nyq], btype="band")
    y = filtfilt(b, a, x, axis=0)
    return np.abs(y)

def window_features(y: np.ndarray) -> dict:
    """Compute simple features for one window (samples x channels)."""
    # MAV, RMS, WL
    mav = np.mean(np.abs(y), axis=0)
    rms = np.sqrt(np.mean(y**2, axis=0))
    wl = np.sum(np.abs(np.diff(y, axis=0)), axis=0)
    return {
        "mav": mav,
        "rms": rms,
        "wl": wl,
    }
