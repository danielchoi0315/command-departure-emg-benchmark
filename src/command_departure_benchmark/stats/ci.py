from __future__ import annotations

from statistics import NormalDist


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score interval for binomial proportions.

    Returns (p_hat, ci_low, ci_high). If n <= 0, returns NaNs.
    """
    n_i = int(n)
    if n_i <= 0:
        nan = float("nan")
        return nan, nan, nan
    k_i = int(k)
    if k_i < 0:
        k_i = 0
    if k_i > n_i:
        k_i = n_i

    p_hat = k_i / n_i
    z = NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / n_i
    center = p_hat + z2 / (2.0 * n_i)
    radius = z * ((p_hat * (1.0 - p_hat) / n_i + z2 / (4.0 * n_i * n_i)) ** 0.5)
    lo = (center - radius) / denom
    hi = (center + radius) / denom
    return float(p_hat), float(max(0.0, lo)), float(min(1.0, hi))
