#!/usr/bin/env bash
set -euo pipefail

if ! command -v aws >/dev/null 2>&1; then
  echo "[DOWNLOAD] ERROR: aws CLI is required but not found."
  exit 2
fi

DATA_ROOT="${DATA_ROOT:-$PWD/lake}"
OUT_GRAB="$DATA_ROOT/raw/physionet_grabmyo"
OUT_HYS="$DATA_ROOT/raw/physionet_hyser"

mkdir -p "$OUT_GRAB" "$OUT_HYS"

echo "[DOWNLOAD] DATA_ROOT=$DATA_ROOT"
echo "[DOWNLOAD] Sync GRABMyo -> $OUT_GRAB/grabmyo-1.0.2/"
aws s3 sync --no-sign-request s3://physionet-open/grabmyo/1.0.2/ "$OUT_GRAB/grabmyo-1.0.2/"

echo "[DOWNLOAD] Sync Hyser -> $OUT_HYS/1.0.0/"
aws s3 sync --no-sign-request s3://physionet-open/hd-semg/1.0.0/ "$OUT_HYS/1.0.0/"

echo "[DOWNLOAD] Complete."
