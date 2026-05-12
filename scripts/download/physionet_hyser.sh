#!/usr/bin/env bash
set -euo pipefail

# Hyser (PhysioNet) downloader.
# Docs: https://physionet.org/content/hd-semg/
# NOTE: PhysioNet may require accepting terms in the browser for some resources.
# This script attempts a direct download of the public files.

DATA_ROOT="${DATA_ROOT:-$PWD/lake}"
OUT="$DATA_ROOT/raw/physionet_hyser"

mkdir -p "$OUT"
cd "$OUT"

# Change VERSION if needed.
VERSION="${1:-1.0.0}"

BASE="https://physionet.org/files/hd-semg/${VERSION}/"

echo "[download] Hyser version=${VERSION} -> ${OUT}"
echo "[download] If this fails due to access policy, download manually from PhysioNet and place files here."

# Recursive fetch (tries to mirror directory)
wget -r -N -c -np --reject "index.html*" "${BASE}" || true

echo "[done] Listing top-level files:"
find . -maxdepth 3 -type f | head
