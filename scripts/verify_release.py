#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED = [
    "README.md",
    "NO_RAW_DATA.md",
    "README_REPRODUCTION.md",
    "CHECKSUMS_SHA256.txt",
    "config/datasets_command_departure.yaml",
    "src/command_departure_benchmark",
    "outputs/tables/db10_primary_table.tex",
    "outputs/tables/exact_budget_summary_table.tex",
    "outputs/tables/db10_extended_risk_boundary_audit_table.tex",
    "outputs/tables/db10_policy_sensitivity_table.tex",
    "outputs/tables/db10_risk_boundary_summary_table.tex",
    "outputs/tables/db10_user_decoder_classical_table.tex",
    "outputs/figures/pdf/fig1_benchmark_flow.pdf",
    "outputs/figures/pdf/fig2_db10_frontier.pdf",
    "outputs/figures/pdf/fig4_exact_budget_audit.pdf",
    "outputs/figures/pdf/figS_db10_context_degradation.pdf",
    "outputs/figures/pdf/figS_db10_calibration_sensitivity.pdf",
    "results/db10_risk_boundary_audit_20260512",
    "results/db10_context_control_audit",
]

FORBIDDEN_TEXT = [
    "D:" + "\\",
    "C:" + "\\",
    "/" + "work" + "/",
    "/" + "scratch" + "/",
    "api" + " key",
    "API" + " key",
    "private compute" + " provider",
    "target" + " journal",
]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".tex", ".yaml", ".yml", ".json", ".csv", ".sh", ".toml", ".cfg"}
CHECKSUM_FILE = "CHECKSUMS_SHA256.txt"


def validate_checksum_paths(root: Path, errors: list[str]) -> None:
    checksum_path = root / CHECKSUM_FILE
    if not checksum_path.exists():
        return
    for line_no, line in enumerate(checksum_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            _, rel_path = line.split(maxsplit=1)
        except ValueError:
            errors.append(f"malformed checksum line {line_no} in {CHECKSUM_FILE}")
            continue
        if "\\" in rel_path:
            errors.append(f"non-portable checksum path on line {line_no}: {rel_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the compact public release package.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()

    errors: list[str] = []
    for rel in REQUIRED:
        if not (root / rel).exists():
            errors.append(f"missing required path: {rel}")

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            errors.append(f"could not read {path.relative_to(root)}: {exc}")
            continue
        for needle in FORBIDDEN_TEXT:
            if needle in text:
                errors.append(f"forbidden text {needle!r} in {path.relative_to(root)}")

    validate_checksum_paths(root, errors)

    if errors:
        print("[VERIFY_RELEASE] FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print("[VERIFY_RELEASE] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
