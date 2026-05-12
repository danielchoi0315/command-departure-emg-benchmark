#!/usr/bin/env bash
set -euo pipefail

ROOT=""
OUT=""
WORKERS="${WORKERS:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$ROOT" ]]; then
  echo "missing --root" >&2
  exit 2
fi
ROOT="$(cd "$ROOT" && pwd)"
OUT="${OUT:-$ROOT/SHA256SUMS.txt}"
if [[ -z "$WORKERS" ]]; then
  WORKERS="$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)"
fi
mkdir -p "$(dirname "$OUT")"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
(
  cd "$ROOT"
  find . -type f ! -name SHA256SUMS.txt -printf '%P\n' | LC_ALL=C sort | xargs -r -d '\n' -P "$WORKERS" sha256sum --
) > "$tmp"
mv "$tmp" "$OUT"
echo "[sha256sum] wrote $OUT"
