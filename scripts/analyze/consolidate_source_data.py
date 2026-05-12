#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _read_first_sheet(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    if not xl.sheet_names:
        return pd.DataFrame()
    return pd.read_excel(path, sheet_name=xl.sheet_names[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidate per-figure and per-table source data into SOURCE_DATA.xlsx.")
    ap.add_argument("--paper_root", type=Path, default=Path("paper"))
    ap.add_argument("--results_root", type=Path, default=Path("results"))
    ap.add_argument("--exp_ids", nargs="*", default=[])
    args = ap.parse_args()

    source_root = args.paper_root / "source_data"
    tables_root = args.paper_root / "tables"
    source_root.mkdir(parents=True, exist_ok=True)
    out = source_root / "SOURCE_DATA.xlsx"

    sheets: dict[str, pd.DataFrame] = {
        "README": pd.DataFrame(
            [
                {"field": "generated_utc", "value": datetime.now(timezone.utc).isoformat()},
                {"field": "paper_root", "value": str(args.paper_root)},
                {"field": "results_root", "value": str(args.results_root)},
                {"field": "exp_ids", "value": ",".join(args.exp_ids)},
            ]
        )
    }
    for workbook in sorted(source_root.glob("*.xlsx")):
        if workbook.name == "SOURCE_DATA.xlsx":
            continue
        sheet = workbook.stem[:31]
        sheets[sheet] = _read_first_sheet(workbook)
    for i, csv_path in enumerate(sorted(tables_root.glob("*.csv")), start=1):
        sheets[f"T{i}"] = pd.read_csv(csv_path)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"[SOURCE_DATA] wrote {out} sheets={list(sheets)}")


if __name__ == "__main__":
    main()
