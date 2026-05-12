#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail-closed lightweight artifact completeness check.")
    ap.add_argument("--artifact_root", type=Path, default=Path("artifacts"))
    ap.add_argument("--require_audit", type=int, default=0)
    ap.add_argument("--require_table_sheets", type=int, default=0)
    ap.add_argument("--min_meta_k", type=int, default=0)
    ap.add_argument("--require_extended_suite", type=int, default=0)
    args = ap.parse_args()

    root = args.artifact_root
    errors: list[str] = []
    if not root.exists():
        raise SystemExit(f"[ARTIFACT_CHECK] missing artifact_root: {root}")
    for sub in ["figures", "source_data", "tables"]:
        if not (root / sub).exists():
            errors.append(f"missing directory: {root / sub}")

    manifest = root / "figure_manifest.json"
    if not manifest.exists():
        errors.append(f"missing figure_manifest.json: {manifest}")
    else:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if not payload:
                errors.append("figure_manifest.json is empty")
        except Exception as exc:
            errors.append(f"invalid figure_manifest.json: {exc}")

    figures = (
        list((root / "figures").glob("*.png")) + list((root / "figures").glob("*.pdf"))
        if (root / "figures").exists()
        else []
    )
    if not figures:
        errors.append(f"no figure PNG/PDF files found under {root / 'figures'}")

    table_csvs = sorted((root / "tables").glob("*.csv")) if (root / "tables").exists() else []
    if not table_csvs:
        errors.append(f"no table CSV files found under {root / 'tables'}")
    for csv_path in table_csvs:
        try:
            if pd.read_csv(csv_path).empty:
                errors.append(f"empty table CSV: {csv_path}")
        except Exception as exc:
            errors.append(f"failed reading table CSV {csv_path}: {exc}")

    source_book = root / "source_data" / "SOURCE_DATA.xlsx"
    if int(args.require_table_sheets) == 1 and not source_book.exists():
        errors.append(f"missing SOURCE_DATA.xlsx: {source_book}")
    if source_book.exists():
        try:
            xl = pd.ExcelFile(source_book)
            if int(args.require_table_sheets) == 1:
                expected = [f"T{i}" for i in range(1, len(table_csvs) + 1)]
                missing = [s for s in expected if s not in xl.sheet_names]
                if missing:
                    errors.append(f"SOURCE_DATA.xlsx missing table sheets: {missing}")
        except Exception as exc:
            errors.append(f"failed reading SOURCE_DATA.xlsx: {exc}")

    if errors:
        print("[ARTIFACT_CHECK] FAIL")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)
    print("[ARTIFACT_CHECK] PASS")


if __name__ == "__main__":
    main()
