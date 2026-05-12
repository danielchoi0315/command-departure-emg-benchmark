from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--datasets_yaml", type=Path, required=True)
    ap.add_argument("--exp_yaml", type=Path, required=True)
    ap.add_argument("--mode", choices=["correctness", "throughput"], default="throughput")
    args = ap.parse_args()

    commands = [
        [
            sys.executable,
            "scripts/train/train_pu.py",
            "--tier",
            str(args.tier),
            "--datasets_yaml",
            str(args.datasets_yaml),
            "--exp_yaml",
            str(args.exp_yaml),
            "--mode",
            args.mode,
        ],
        [
            sys.executable,
            "scripts/train/train_workload.py",
            "--tier",
            str(args.tier),
            "--datasets_yaml",
            str(args.datasets_yaml),
            "--exp_yaml",
            str(args.exp_yaml),
        ],
        [
            sys.executable,
            "scripts/train/train_pa.py",
            "--tier",
            str(args.tier),
            "--datasets_yaml",
            str(args.datasets_yaml),
            "--exp_yaml",
            str(args.exp_yaml),
        ],
    ]
    for cmd in commands:
        subprocess.run(cmd, check=True, env=os.environ.copy())

    print("[TRAIN] Completed p_u, workload, and p_a training stages.")


if __name__ == "__main__":
    main()
