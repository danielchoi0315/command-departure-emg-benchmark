from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AutoBatchResult:
    batch_size: int
    mode: str
    strategy: str
    device: str
    target_utilization: float
    max_batch_tested: int
    oom_observed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": int(self.batch_size),
            "mode": self.mode,
            "strategy": self.strategy,
            "device": self.device,
            "target_utilization": float(self.target_utilization),
            "max_batch_tested": int(self.max_batch_tested),
            "oom_observed": bool(self.oom_observed),
        }


def _build_probe_batch(sample_batch: torch.Tensor, batch_size: int) -> torch.Tensor:
    base = int(sample_batch.shape[0])
    if base <= 0:
        raise ValueError("sample_batch must have batch dimension > 0")
    reps = (batch_size + base - 1) // base
    tiled = sample_batch.repeat((reps,) + (1,) * (sample_batch.dim() - 1))
    return tiled[:batch_size]


def _try_forward(
    model: torch.nn.Module,
    batch: torch.Tensor,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
    device: str,
) -> tuple[bool, bool]:
    oom = False
    try:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _ = model(batch)
        return True, oom
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            oom = True
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            return False, oom
        raise


def autobatch_size(
    model: torch.nn.Module,
    sample_batch: torch.Tensor,
    *,
    mode: str = "throughput",
    requested_batch_size: int | str | None = "auto_max_vram",
    target_utilization: float = 0.85,
    min_batch: int = 16,
    correctness_batch: int = 64,
    cpu_default_batch: int = 256,
    max_batch_cap: int = 16384,
) -> AutoBatchResult:
    if isinstance(requested_batch_size, int):
        return AutoBatchResult(
            batch_size=max(1, int(requested_batch_size)),
            mode=mode,
            strategy="fixed",
            device="cuda" if torch.cuda.is_available() else "cpu",
            target_utilization=target_utilization,
            max_batch_tested=max(1, int(requested_batch_size)),
            oom_observed=False,
        )

    if mode == "correctness":
        return AutoBatchResult(
            batch_size=max(1, int(correctness_batch)),
            mode=mode,
            strategy="correctness_conservative",
            device="cuda" if torch.cuda.is_available() else "cpu",
            target_utilization=target_utilization,
            max_batch_tested=max(1, int(correctness_batch)),
            oom_observed=False,
        )

    if requested_batch_size not in {None, "auto_max_vram"}:
        try:
            fixed = int(requested_batch_size)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid requested_batch_size={requested_batch_size}") from exc
        return AutoBatchResult(
            batch_size=max(1, fixed),
            mode=mode,
            strategy="fixed",
            device="cuda" if torch.cuda.is_available() else "cpu",
            target_utilization=target_utilization,
            max_batch_tested=max(1, fixed),
            oom_observed=False,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        return AutoBatchResult(
            batch_size=max(1, int(cpu_default_batch)),
            mode=mode,
            strategy="cpu_default",
            device=device,
            target_utilization=target_utilization,
            max_batch_tested=max(1, int(cpu_default_batch)),
            oom_observed=False,
        )

    model = model.to(device)
    model.eval()
    amp_dtype = torch.bfloat16
    use_amp = True

    total_mem = int(torch.cuda.get_device_properties(0).total_memory)
    target_mem = int(total_mem * max(0.10, min(0.98, target_utilization)))

    low = max(1, min_batch)
    high = low
    best = low
    max_tested = low
    saw_oom = False

    # Exponential growth phase.
    while high <= max_batch_cap:
        probe = _build_probe_batch(sample_batch, high).to(device, non_blocking=True)
        ok, oom = _try_forward(model, probe, use_amp=use_amp, amp_dtype=amp_dtype, device=device)
        max_tested = max(max_tested, high)
        saw_oom = saw_oom or oom
        if not ok:
            break
        best = high
        used = int(torch.cuda.max_memory_allocated())
        if used >= target_mem:
            break
        high *= 2

    if high > max_batch_cap:
        return AutoBatchResult(
            batch_size=best,
            mode=mode,
            strategy="autotuned_max_cap",
            device=device,
            target_utilization=target_utilization,
            max_batch_tested=max_tested,
            oom_observed=saw_oom,
        )

    # Binary search between known-good `best` and failing/ceiling `high`.
    lo = best
    hi = max(best + 1, high)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        probe = _build_probe_batch(sample_batch, mid).to(device, non_blocking=True)
        ok, oom = _try_forward(model, probe, use_amp=use_amp, amp_dtype=amp_dtype, device=device)
        max_tested = max(max_tested, mid)
        saw_oom = saw_oom or oom
        if ok:
            lo = mid
        else:
            hi = mid

    return AutoBatchResult(
        batch_size=max(min_batch, lo),
        mode=mode,
        strategy="autotuned_binary_search",
        device=device,
        target_utilization=target_utilization,
        max_batch_tested=max_tested,
        oom_observed=saw_oom,
    )


def make_autobatch_record(result: AutoBatchResult, *, model_name: str, fold: int, dataset: str) -> dict[str, Any]:
    out = result.as_dict()
    out.update({"model": model_name, "fold": int(fold), "dataset": dataset})
    return out
