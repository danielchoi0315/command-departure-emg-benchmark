from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _quantile_linear(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q=q, method="linear"))
    except TypeError:
        # NumPy<1.22 compatibility.
        return float(np.quantile(values, q=q, interpolation="linear"))


def _clip_rate(rate: float, r_min: float = 0.0, r_max: float = 1.0) -> float:
    return float(np.clip(float(rate), float(r_min), float(r_max)))


def _quantile_level(target_gate_rate: float, *, use_complement_quantile: bool) -> float:
    r = _clip_rate(target_gate_rate, 0.0, 1.0)
    q = 1.0 - r if use_complement_quantile else r
    return _clip_rate(q, 0.0, 1.0)


def fit_tau_from_train_confidence(
    c_u_train: np.ndarray,
    target_gate_rate: float,
    *,
    use_complement_quantile: bool = False,
) -> float:
    """Fit tau from training confidence distribution only.

    Gate definition: trigger when `c_u < tau`.
    To avoid strict-inequality edge cases on discrete confidence supports,
    we move the chosen quantile to the next representable float toward +inf.
    """
    c = np.asarray(c_u_train, dtype=float)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return float("nan")
    q = _quantile_level(target_gate_rate, use_complement_quantile=use_complement_quantile)
    tau = _quantile_linear(c, q=q)
    return float(np.nextafter(tau, np.inf))


@dataclass(frozen=True)
class DynamicGateRateParams:
    a: float
    b_workload: float
    c_entropy: float
    r_min: float = 0.05
    r_max: float = 0.30


def target_gate_rate_dyn(workload: np.ndarray, entropy: np.ndarray, params: DynamicGateRateParams) -> np.ndarray:
    w = np.asarray(workload, dtype=float)
    h = np.asarray(entropy, dtype=float)
    logits = params.a + params.b_workload * w + params.c_entropy * h
    rates = 1.0 / (1.0 + np.exp(-logits))
    return np.clip(rates, params.r_min, params.r_max).astype(float)


def calibrate_budgeted_dynamic_rates(
    train_workload: np.ndarray,
    train_entropy: np.ndarray,
    test_workload: np.ndarray,
    test_entropy: np.ndarray,
    params: DynamicGateRateParams,
    target_mean_gate_rate: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Shift dynamic rates so train-fold mean matches a target gate rate.

    Uses only train-fold workload/entropy to fit a scalar shift and applies
    it to test-fold rates. No labels are used.
    """
    train_base = target_gate_rate_dyn(train_workload, train_entropy, params)
    train_mean = float(np.mean(train_base)) if len(train_base) else float("nan")
    target = _clip_rate(float(target_mean_gate_rate), params.r_min, params.r_max)
    shift = float(target - train_mean) if np.isfinite(train_mean) else 0.0

    test_base = target_gate_rate_dyn(test_workload, test_entropy, params)
    test_budget = np.clip(test_base + shift, params.r_min, params.r_max).astype(float)
    train_budget = np.clip(train_base + shift, params.r_min, params.r_max).astype(float)
    return test_budget, {
        "target_mean_gate_rate": target,
        "train_base_mean": train_mean,
        "train_budget_mean": float(np.mean(train_budget)) if len(train_budget) else float("nan"),
        "shift": shift,
    }
