from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommandDepartureWindow:
    """Canonical per-decision-window record (dataset agnostic).

    This mirrors the benchmark logging fields:
      p_u, p_a, g_u, g_a, g*, c_u, CDC, tau

    Notes:
    - p_u and p_a are probability vectors over dataset-native K classes.
    - g_u, g_a, and g_star are integer class indices in [0, K-1].
    """

    dataset: str
    subject_id: str
    session_id: str
    trial_id: str
    window_id: int
    t_start: float
    t_end: float
    g_star: int

    p_u: np.ndarray
    p_a: np.ndarray

    # Derived values.
    g_u: int
    g_a: int
    c_u: float
    H_pu: float
    H_pa: float

    # Policy logs.
    tau_set: float
    tau_dyn: float
    cdc: float

    g_hat_useronly: int
    g_hat_confblend: int
    g_hat_setacsa: int
    g_hat_csaab: int


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def argmax_int(p: np.ndarray) -> int:
    return int(np.argmax(p))
