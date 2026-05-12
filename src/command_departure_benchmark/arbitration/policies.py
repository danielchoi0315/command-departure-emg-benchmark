from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from command_departure_benchmark.schema import entropy, argmax_int

def confblend(p_u: np.ndarray, p_a: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    p = alpha * p_u + (1.0 - alpha) * p_a
    p = np.clip(p, 1e-12, None)
    return p / p.sum()

def gate_by_tau(p_u: np.ndarray, p_a: np.ndarray, tau: float) -> int:
    """Simple τ-gating: if user confidence >= τ => take g_u else take g_a."""
    g_u = argmax_int(p_u)
    g_a = argmax_int(p_a)
    c_u = float(np.max(p_u))
    return g_u if c_u >= float(tau) else g_a

@dataclass(frozen=True)
class CSAABParams:
    tau_min: float
    tau_max: float
    base_tau: float
    k_workload: float
    k_entropy: float
    entropy_ref: float
    workload_ref: float

def csaab_tau(params: CSAABParams, workload: float, H_pu: float) -> float:
    """Simple monotone τ(t)=base + k_w*(w-w0) + k_h*(H-H0), clamped."""
    tau = (
        params.base_tau
        + params.k_workload * (float(workload) - params.workload_ref)
        + params.k_entropy * (float(H_pu) - params.entropy_ref)
    )
    return float(np.clip(tau, params.tau_min, params.tau_max))
