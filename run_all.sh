#!/usr/bin/env bash
set -euo pipefail

TIER="0"
EXP="command_departure_template"
PAPER_MODE="0"
EXP_MODE=""
DATASETS_YAML="config/datasets.yaml"
EXP_YAML="config/experiment.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --exp)  EXP="$2"; shift 2 ;;
    --paper_mode) PAPER_MODE="$2"; shift 2 ;;
    --exp_mode) EXP_MODE="$2"; shift 2 ;;
    --datasets_yaml) DATASETS_YAML="$2"; shift 2 ;;
    --exp_yaml) EXP_YAML="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Auto-select supplementary GPU experiment config when using *_gpu exp IDs
# and caller did not explicitly pass --exp_yaml.
if [[ "$EXP_YAML" == "config/experiment.yaml" && "$EXP" == *_gpu && -f "config/experiment_gpu.yaml" ]]; then
  EXP_YAML="config/experiment_gpu.yaml"
fi

export EXP_ID="$EXP"
export TIER="$TIER"
export EXP_MODE="$EXP_MODE"
export DATA_ROOT="${DATA_ROOT:-$PWD/lake}"
if [[ -z "${PAPER_ROOT:-}" ]]; then
  PAPER_ROOT="artifacts/$EXP_ID"
fi
export PAPER_ROOT
RESULTS_ROOT="results/$EXP"
ENV_PATH="$RESULTS_ROOT/ENV.txt"
STAGE_MANIFEST_DIR="$RESULTS_ROOT/manifests"
SKIP_PREPROCESS="${COMMAND_DEPARTURE_SKIP_PREPROCESS:-0}"
E2E_PAPER_MODE="0"
if [[ "$PAPER_MODE" == "1" ]]; then
  E2E_PAPER_MODE="1"
fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x "$PWD/.venv/bin/python" ]]; then
  PYTHON_CMD="$PWD/.venv/bin/python"
else
  PYTHON_CMD="python"
fi
if [[ -d "$PWD/src" ]]; then
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
  else
    export PYTHONPATH="$PWD/src"
  fi
fi

if [[ -z "$EXP_MODE" ]]; then
  EXP_MODE="$("$PYTHON_CMD" - "$EXP_YAML" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
exp = cfg.get("experiment", {})
print(exp.get("exp_mode", "throughput"))
PY
)"
fi
if [[ "$EXP_MODE" != "throughput" && "$EXP_MODE" != "correctness" ]]; then
  echo "[CommandDepartureBenchmark] Invalid EXP_MODE=$EXP_MODE (expected throughput|correctness)"
  exit 2
fi

mkdir -p "$RESULTS_ROOT"
mkdir -p "$STAGE_MANIFEST_DIR"
"$PYTHON_CMD" scripts/utils/env_report.py --fs-target "$PWD" --repo-hint "$PWD" > "$ENV_PATH"

echo "[CommandDepartureBenchmark] PYTHON_CMD=$PYTHON_CMD  DATA_ROOT=$DATA_ROOT  EXP_ID=$EXP_ID  TIER=$TIER  PAPER_MODE=$PAPER_MODE  EXP_MODE=$EXP_MODE  PAPER_ROOT=$PAPER_ROOT"
echo "[CommandDepartureBenchmark] Environment report: $ENV_PATH"

if [[ "$PAPER_MODE" == "1" ]]; then
  "$PYTHON_CMD" - "$DATASETS_YAML" "$DATA_ROOT" <<'PY'
import importlib
import sys
from pathlib import Path

import yaml

datasets_yaml = Path(sys.argv[1])
data_root = Path(sys.argv[2])
datasets = yaml.safe_load(datasets_yaml.read_text())["datasets"]

tier0 = []
missing = []
for ds_id, info in datasets.items():
    if int(info.get("tier", 99)) != 0:
        continue
    tier0.append(ds_id)
    raw_path = data_root / info["raw_subdir"]
    try:
        mod = importlib.import_module(f"command_departure_benchmark.adapters.{ds_id}")
    except Exception as exc:
        missing.append((ds_id, str(raw_path), f"adapter import failed: {exc}"))
        continue
    try:
        ok = bool(mod.available(raw_path))
    except Exception as exc:
        missing.append((ds_id, str(raw_path), f"adapter available() error: {exc}"))
        continue
    if not ok:
        missing.append((ds_id, str(raw_path), "adapter available() returned false"))

if not tier0:
    print("[PAPER_MODE] FAIL: no tier0 datasets declared in datasets.yaml")
    raise SystemExit(2)

if missing:
    print("[PAPER_MODE] FAIL: required tier0 datasets are not adapter-available:")
    for ds_id, raw_path, reason in missing:
        print(f"  - {ds_id}: expected at {raw_path} ({reason})")
    raise SystemExit(2)

print(f"[PAPER_MODE] PASS: all required tier0 datasets are present ({len(tier0)} datasets).")
PY
fi

# 1) QC + preprocess (requires adapters)
PREPROCESS_START="$(date --iso-8601=seconds)"
if [[ "$SKIP_PREPROCESS" == "1" ]]; then
  echo "[CommandDepartureBenchmark] Skipping adapter preprocessing (COMMAND_DEPARTURE_SKIP_PREPROCESS=1); validating existing derived windows."
else
  "$PYTHON_CMD" scripts/preprocess/run_all_adapters.py --tier "$TIER" --datasets_yaml "$DATASETS_YAML" --exp_yaml "$EXP_YAML"
fi
"$PYTHON_CMD" scripts/preprocess/validate_derived_schema.py --tier "$TIER" --datasets_yaml "$DATASETS_YAML" --data_root "$DATA_ROOT"
PREPROCESS_END="$(date --iso-8601=seconds)"
"$PYTHON_CMD" scripts/utils/write_manifest.py \
  --stage preprocess \
  --out "$STAGE_MANIFEST_DIR/preprocess_manifest.json" \
  --input "$DATASETS_YAML" \
  --output "$DATA_ROOT/derived" \
  --config "$DATASETS_YAML" \
  --env-file "$ENV_PATH" \
  --repo-hint "$PWD" \
  --stage-start "$PREPROCESS_START" \
  --stage-end "$PREPROCESS_END"

# 2) Train p_u, w(t), p_a models (stubs; dataset adapters must provide inputs)
TRAIN_START="$(date --iso-8601=seconds)"
"$PYTHON_CMD" scripts/train/train_all.py --tier "$TIER" --datasets_yaml "$DATASETS_YAML" --exp_yaml "$EXP_YAML" --mode "$EXP_MODE"
TRAIN_END="$(date --iso-8601=seconds)"
"$PYTHON_CMD" scripts/utils/write_manifest.py \
  --stage train \
  --out "$STAGE_MANIFEST_DIR/train_manifest.json" \
  --input "$EXP_YAML" \
  --output "$RESULTS_ROOT" \
  --config "$EXP_YAML" \
  --env-file "$ENV_PATH" \
  --repo-hint "$PWD" \
  --stage-start "$TRAIN_START" \
  --stage-end "$TRAIN_END"

# 2b) Fail-closed p_a oracle audit before simulation.
"$PYTHON_CMD" scripts/audit/pa_oracle_audit.py \
  --exp "$EXP_ID" \
  --tier "$TIER" \
  --datasets_yaml "$DATASETS_YAML" \
  --results_root "results" \
  --data_root "$DATA_ROOT"

# 3) Simulate arbitration
SIM_START="$(date --iso-8601=seconds)"
"$PYTHON_CMD" scripts/simulate/simulate_all.py --tier "$TIER" --datasets_yaml "$DATASETS_YAML" --exp_yaml "$EXP_YAML" --paper_mode "$PAPER_MODE"
SIM_END="$(date --iso-8601=seconds)"
"$PYTHON_CMD" scripts/utils/write_manifest.py \
  --stage simulate \
  --out "$STAGE_MANIFEST_DIR/simulate_manifest.json" \
  --input "$EXP_YAML" \
  --output "$RESULTS_ROOT" \
  --config "$EXP_YAML" \
  --env-file "$ENV_PATH" \
  --repo-hint "$PWD" \
  --stage-start "$SIM_START" \
  --stage-end "$SIM_END"

# 4) Aggregate + meta-analysis + figures + source data export
ANALYZE_START="$(date --iso-8601=seconds)"
"$PYTHON_CMD" scripts/analyze/make_paper_artifacts.py --tier "$TIER" --datasets_yaml "$DATASETS_YAML" --exp_yaml "$EXP_YAML" --paper_root "$PAPER_ROOT"
if [[ "$E2E_PAPER_MODE" == "1" ]]; then
  "$PYTHON_CMD" scripts/audit/command_departure_audit.py --exp "$EXP_ID" --tier "$TIER" --datasets_yaml "$DATASETS_YAML" --paper_root "$PAPER_ROOT"
fi
"$PYTHON_CMD" scripts/analyze/consolidate_source_data.py --paper_root "$PAPER_ROOT" --results_root "results" --exp_ids "$EXP_ID"
CHECK_ARGS=(--artifact_root "$PAPER_ROOT")
if [[ "$E2E_PAPER_MODE" == "1" ]]; then
  CHECK_ARGS+=(--require_audit 1 --require_table_sheets 1)
  CHECK_ARGS+=(--min_meta_k 3 --require_extended_suite 1)
fi
"$PYTHON_CMD" scripts/utils/check_artifact_completeness.py "${CHECK_ARGS[@]}"
ANALYZE_END="$(date --iso-8601=seconds)"
"$PYTHON_CMD" scripts/utils/write_manifest.py \
  --stage analyze \
  --out "$STAGE_MANIFEST_DIR/analyze_manifest.json" \
  --input "$EXP_YAML" \
  --output "$PAPER_ROOT" \
  --config "$EXP_YAML" \
  --env-file "$ENV_PATH" \
  --repo-hint "$PWD" \
  --stage-start "$ANALYZE_START" \
  --stage-end "$ANALYZE_END"

"$PYTHON_CMD" scripts/utils/write_manifest.py \
  --stage run_all \
  --out "$RESULTS_ROOT/manifest.json" \
  --input "$DATASETS_YAML" \
  --input "$EXP_YAML" \
  --output "$RESULTS_ROOT" \
  --output "$PAPER_ROOT" \
  --config "$DATASETS_YAML" \
  --config "$EXP_YAML" \
  --env-file "$ENV_PATH" \
  --repo-hint "$PWD"

echo "[CommandDepartureBenchmark] Done. See $RESULTS_ROOT and $PAPER_ROOT/."
