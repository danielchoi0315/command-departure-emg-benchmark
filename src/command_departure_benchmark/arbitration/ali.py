from __future__ import annotations

import numpy as np


def ali_from_tau(tau: float) -> float:
    """Backward-compatible command-departure-cost helper.

    The public benchmark reports decoder-relative command-departure cost (CDC).
    The legacy function name is retained so older scripts continue to run, but
    paper-facing outputs should use CDC terminology.
    """
    tau = float(np.clip(tau, 0.0, 1.0))
    return 1.0 - tau
