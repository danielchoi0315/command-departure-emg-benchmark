#!/usr/bin/env bash
set -euo pipefail

# GRABMyo (PhysioNet) downloader.
# Docs: https://physionet.org/content/grabmyo/

DATA_ROOT="${DATA_ROOT:-$PWD/lake}"
OUT="$DATA_ROOT/raw/physionet_grabmyo"

mkdir -p "$OUT"
cd "$OUT"

VERSION="${1:-1.0.0}"
BASE="https://physionet.org/files/grabmyo/${VERSION}/"

echo "[download] GRABMyo version=${VERSION} -> ${OUT}"
echo "[download] If this fails due to access policy, download manually from PhysioNet and place files here."

wget -r -N -c -np --reject "index.html*" "${BASE}" || true

echo "[done] Listing top-level files:"
find . -maxdepth 3 -type f | head
