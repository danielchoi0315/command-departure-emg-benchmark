#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _bytes_to_gib(num_bytes: int) -> float:
    return round(num_bytes / (1024 ** 3), 2)


def _read_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return platform.processor() or "unknown"
    for line in cpuinfo.read_text().splitlines():
        if line.lower().startswith("model name"):
            _, _, value = line.partition(":")
            return value.strip()
    return platform.processor() or "unknown"


def _read_mem_total_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                # /proc/meminfo reports kB.
                return int(parts[1]) * 1024
    return None


def _gather_torch() -> dict[str, Any]:
    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_version": None,
        "cudnn_version": None,
        "cuda_available": False,
        "gpus": [],
        "error": None,
    }
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime env
        info["error"] = str(exc)
        return info

    info["installed"] = True
    info["version"] = getattr(torch, "__version__", None)
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    cudnn_version = None
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None
    info["cudnn_version"] = cudnn_version
    info["cuda_available"] = bool(torch.cuda.is_available())

    gpus: list[dict[str, Any]] = []
    if info["cuda_available"]:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_vram_bytes": int(props.total_memory),
                    "total_vram_gib": _bytes_to_gib(int(props.total_memory)),
                }
            )
    info["gpus"] = gpus
    return info


def _gather_nvidia_smi() -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "summary": [],
        "cuda_version": None,
        "error": None,
    }
    if shutil.which("nvidia-smi") is None:
        out["error"] = "nvidia-smi not found"
        return out

    rc, full_stdout, full_stderr = _run(["nvidia-smi"])
    if rc != 0:
        out["error"] = full_stderr or "nvidia-smi failed"
        return out

    for line in full_stdout.splitlines():
        if "CUDA Version:" in line:
            rhs = line.split("CUDA Version:", 1)[1].strip()
            out["cuda_version"] = rhs.split()[0]
            break

    rc, stdout, stderr = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if rc != 0:
        out["error"] = stderr or "query failed"
        return out

    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_total": parts[2],
                "driver_version": parts[3],
                "cuda_version": out["cuda_version"],
            }
        )
    out["available"] = True
    out["summary"] = rows
    return out


def _gather_git(repo_hint: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "available": False,
        "repo_root": None,
        "commit": None,
        "dirty": None,
        "error": None,
    }
    if shutil.which("git") is None:
        info["error"] = "git not found"
        return info

    rc, root, err = _run(["git", "rev-parse", "--show-toplevel"], cwd=repo_hint)
    if rc != 0:
        info["error"] = err or "not in a git repository"
        return info

    repo_root = Path(root)
    rc, commit, err = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if rc != 0:
        info["error"] = err or "failed to resolve HEAD"
        return info

    rc, status, err = _run(["git", "status", "--porcelain"], cwd=repo_root)
    if rc != 0:
        info["error"] = err or "failed to read git status"
        return info

    info["available"] = True
    info["repo_root"] = str(repo_root)
    info["commit"] = commit
    info["dirty"] = bool(status.strip())
    return info


def collect_report(target_path: Path, repo_hint: Path) -> dict[str, Any]:
    fs_usage = shutil.disk_usage(target_path)
    cpu_model = _read_cpu_model()
    ram_total = _read_mem_total_bytes()
    torch_info = _gather_torch()
    nvidia_info = _gather_nvidia_smi()
    git_info = _gather_git(repo_hint)

    return {
        "generated_at_utc": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": str(Path.cwd()),
        "nvidia_smi": nvidia_info,
        "torch": torch_info,
        "gpu": {
            "name": torch_info["gpus"][0]["name"] if torch_info["gpus"] else None,
            "total_vram_gib": torch_info["gpus"][0]["total_vram_gib"]
            if torch_info["gpus"]
            else None,
        },
        "cpu": {
            "model": cpu_model,
            "core_count_logical": os.cpu_count(),
            "ram_total_bytes": ram_total,
            "ram_total_gib": _bytes_to_gib(ram_total) if ram_total else None,
        },
        "filesystem": {
            "target": str(target_path),
            "total_bytes": fs_usage.total,
            "used_bytes": fs_usage.used,
            "free_bytes": fs_usage.free,
            "free_gib": _bytes_to_gib(fs_usage.free),
        },
        "git": git_info,
    }


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=== CommandDepartureBenchmark Environment Report ===")
    lines.append(f"generated_at_utc: {report['generated_at_utc']}")
    lines.append(f"hostname: {report['hostname']}")
    lines.append(f"platform: {report['platform']}")
    lines.append(f"python: {report['python']}")
    lines.append(f"cwd: {report['cwd']}")
    lines.append("")
    lines.append("== nvidia-smi summary ==")
    nvidia = report["nvidia_smi"]
    if nvidia["available"] and nvidia["summary"]:
        for row in nvidia["summary"]:
            lines.append(
                f"gpu[{row['index']}]: {row['name']} | total={row['memory_total']} | "
                f"driver={row['driver_version']} | cuda={row['cuda_version']}"
            )
    else:
        lines.append(f"unavailable: {nvidia['error']}")
    lines.append("")
    lines.append("== torch/cuda/cudnn ==")
    torch_info = report["torch"]
    if torch_info["installed"]:
        lines.append(f"torch: {torch_info['version']}")
        lines.append(f"torch.cuda: {torch_info['cuda_version']}")
        lines.append(f"torch.cudnn: {torch_info['cudnn_version']}")
        lines.append(f"cuda_available: {torch_info['cuda_available']}")
    else:
        lines.append(f"torch unavailable: {torch_info['error']}")
    lines.append("")
    lines.append("== gpu summary ==")
    gpu = report["gpu"]
    lines.append(f"name: {gpu['name']}")
    lines.append(f"total_vram_gib: {gpu['total_vram_gib']}")
    lines.append("")
    lines.append("== cpu/ram ==")
    cpu = report["cpu"]
    lines.append(f"model: {cpu['model']}")
    lines.append(f"logical_cores: {cpu['core_count_logical']}")
    lines.append(f"ram_total_gib: {cpu['ram_total_gib']}")
    lines.append("")
    lines.append("== filesystem ==")
    fs = report["filesystem"]
    lines.append(f"target: {fs['target']}")
    lines.append(f"free_gib: {fs['free_gib']}")
    lines.append("")
    lines.append("== git ==")
    git = report["git"]
    if git["available"]:
        lines.append(f"repo_root: {git['repo_root']}")
        lines.append(f"commit: {git['commit']}")
        lines.append(f"dirty: {git['dirty']}")
    else:
        lines.append(f"unavailable: {git['error']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a hardware/software environment report.")
    parser.add_argument(
        "--fs-target",
        type=Path,
        default=Path.cwd(),
        help="Path used for filesystem free-space reporting (default: cwd).",
    )
    parser.add_argument(
        "--repo-hint",
        type=Path,
        default=Path.cwd(),
        help="Directory to probe for git metadata (default: cwd).",
    )
    parser.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional path to also write machine-readable JSON.",
    )
    args = parser.parse_args()

    report = collect_report(args.fs_target, args.repo_hint)
    text = render_text(report)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
