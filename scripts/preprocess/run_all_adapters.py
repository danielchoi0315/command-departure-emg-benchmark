from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import importlib
from tqdm import tqdm

def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True, help="Max tier to include, e.g. 0 includes only tier0, 1 includes tiers 0-1.")
    ap.add_argument("--datasets_yaml", type=Path, required=True)
    ap.add_argument("--exp_yaml", type=Path, default=None, help="Optional experiment config to provide window/hop to adapters.")
    args = ap.parse_args()

    dsreg = load_yaml(args.datasets_yaml)["datasets"]
    exp_window = {}
    if args.exp_yaml is not None and args.exp_yaml.exists():
        try:
            exp_cfg = load_yaml(args.exp_yaml).get("experiment", {})
            exp_window = dict(exp_cfg.get("window", {}) or {})
        except Exception:
            exp_window = {}
    data_root = Path(__import__("os").environ.get("DATA_ROOT", "lake"))
    raw_root = data_root
    derived_root = data_root / "derived"

    for ds_id, info in tqdm(dsreg.items(), desc="datasets"):
        if int(info.get("tier", 99)) > args.tier:
            continue
        adapter_module = info.get("adapter_module", ds_id)
        try:
            mod = importlib.import_module(f"command_departure_benchmark.adapters.{adapter_module}")
        except ModuleNotFoundError:
            print(
                f"[SKIP] {ds_id}: no adapter module "
                f"command_departure_benchmark.adapters.{adapter_module} (status={info.get('adapter_status', 'unknown')})."
            )
            continue
        ds_raw = raw_root / info["raw_subdir"]
        ds_out = derived_root / ds_id / info.get("version", "unknown")
        ds_out.mkdir(parents=True, exist_ok=True)
        adapter_cfg = dict(info)
        if exp_window:
            adapter_cfg["window"] = dict(exp_window)
            if "seconds" in exp_window:
                adapter_cfg["window_seconds"] = float(exp_window["seconds"])
            if "hop_seconds" in exp_window:
                adapter_cfg["hop_seconds"] = float(exp_window["hop_seconds"])

        if not mod.available(ds_raw):
            print(f"[SKIP] {ds_id}: raw data not found at {ds_raw}. (access={info.get('access')})")
            continue

        try:
            out_path = mod.preprocess(ds_raw, ds_out, cfg=adapter_cfg)
        except Exception as exc:
            print(f"[SKIP] {ds_id}: preprocess failed ({exc})")
            continue
        print(f"[OK] {ds_id}: wrote {out_path}")

if __name__ == "__main__":
    main()
