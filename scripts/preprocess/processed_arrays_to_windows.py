#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DATASET_MAP = {
    "ninapro_db10_meganepro": {
        "processed_name": "db10",
        "version": "2020",
        "source_name": "DB10/MeganePro",
    },
    "physionet_grabmyo": {
        "processed_name": "grabmyo",
        "version": "1.0.2",
        "source_name": "GRABMyo",
    },
    "physionet_hyser": {
        "processed_name": "hyser",
        "version": "1.0.0",
        "source_name": "Hyser",
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _as_list_column(x: np.ndarray) -> list[list[float]]:
    if x.ndim != 2:
        raise ValueError(f"expected 2D feature matrix, got shape={x.shape}")
    return [row.astype(float, copy=False).tolist() for row in x]


def _stable_window_ids(meta: pd.DataFrame) -> pd.Series:
    if "episode_id" in meta.columns and "prefix_time_s" in meta.columns:
        return meta["episode_id"].astype(str) + "_" + meta["prefix_time_s"].astype(str)
    if "record_id" in meta.columns and "timestamp_s" in meta.columns:
        return meta["record_id"].astype(str) + "_" + meta["timestamp_s"].astype(str)
    return pd.Series(np.arange(len(meta)).astype(str), index=meta.index)


def _load_processed_dataset(src: Path, *, ds_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = ["metadata.csv", "user_features.npy", "assist_features.npy", "labels.npy", "label_vocab.npy"]
    missing = [name for name in required if not (src / name).exists()]
    if missing:
        raise FileNotFoundError(f"{src}: missing required processed files: {missing}")

    meta = pd.read_csv(src / "metadata.csv")
    user = np.load(src / "user_features.npy", mmap_mode="r")
    assist = np.load(src / "assist_features.npy", mmap_mode="r")
    labels = np.load(src / "labels.npy", mmap_mode="r")
    vocab = np.load(src / "label_vocab.npy", allow_pickle=True)

    n = len(meta)
    if user.shape[0] != n or assist.shape[0] != n or labels.shape[0] != n:
        raise ValueError(
            f"{src}: row mismatch metadata={n} user={user.shape[0]} assist={assist.shape[0]} labels={labels.shape[0]}"
        )

    out = pd.DataFrame()
    out["dataset"] = ds_id
    out["subject_id"] = meta.get("subject_id", pd.Series(["unknown"] * n)).astype(str).to_numpy()
    out["session_id"] = meta.get("session", meta.get("day", pd.Series(["session0"] * n))).astype(str).to_numpy()
    out["trial_id"] = meta.get("episode_id", meta.get("record_id", pd.Series(np.arange(n)))).astype(str).to_numpy()
    out["window_id"] = _stable_window_ids(meta).astype(str).to_numpy()
    out["g_star"] = labels.astype(int)
    out["X_pu"] = _as_list_column(np.asarray(user))
    out["X_pa"] = _as_list_column(np.asarray(assist))

    # The current workload model is optional. Supplying neutral workload keeps
    # CSAAB deterministic without making a human-workload claim.
    out["workload"] = 0.5
    out["processed_source_dataset"] = DATASET_MAP[ds_id]["processed_name"]
    if "record_id" in meta.columns:
        out["source_record"] = meta["record_id"].astype(str).to_numpy()
    elif "episode_id" in meta.columns:
        out["source_record"] = meta["episode_id"].astype(str).to_numpy()
    else:
        out["source_record"] = ""

    manifest = {
        "dataset": ds_id,
        "source_dir": str(src),
        "n_rows": int(n),
        "n_subjects": int(out["subject_id"].nunique()),
        "n_classes": int(np.unique(labels).size),
        "user_feature_dim": int(user.shape[1]),
        "assist_feature_dim": int(assist.shape[1]),
        "label_vocab": [str(v) for v in vocab.tolist()],
        "columns": list(out.columns),
    }
    return out, manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert local processed EMG arrays into CommandDepartureBenchmark windows.parquet files.")
    ap.add_argument("--processed_root", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--datasets_yaml", type=Path, default=Path("config/datasets_command_departure.yaml"))
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--datasets", nargs="*", default=None, help="Optional dataset ids to convert.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    registry = _load_yaml(args.datasets_yaml)["datasets"]
    selected = set(args.datasets) if args.datasets else set(DATASET_MAP)
    reports: list[dict[str, Any]] = []

    for ds_id, info in registry.items():
        if int(info.get("tier", 99)) > args.tier or ds_id not in selected or ds_id not in DATASET_MAP:
            continue
        source = args.processed_root / DATASET_MAP[ds_id]["processed_name"]
        out_dir = args.data_root / "derived" / ds_id / str(info.get("version", DATASET_MAP[ds_id]["version"]))
        out_path = out_dir / "windows.parquet"
        manifest_path = out_dir / "processed_arrays_manifest.json"
        if out_path.exists() and not args.overwrite:
            reports.append({"dataset": ds_id, "status": "skipped_existing", "windows": str(out_path)})
            continue

        windows, manifest = _load_processed_dataset(source, ds_id=ds_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".parquet.tmp")
        windows.to_parquet(tmp_path, index=False)
        tmp_path.replace(out_path)
        manifest["status"] = "ok"
        manifest["windows"] = str(out_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        reports.append(manifest)
        print(f"[PROCESSED_TO_WINDOWS] {ds_id}: wrote {out_path} rows={len(windows)}")

    report_path = args.data_root / "derived" / "processed_arrays_conversion_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"processed_root": str(args.processed_root), "datasets": reports}, indent=2), encoding="utf-8")
    if not reports:
        raise SystemExit("No datasets converted. Check --processed_root, --datasets_yaml, and --tier.")
    print(f"[PROCESSED_TO_WINDOWS] report: {report_path}")


if __name__ == "__main__":
    main()
