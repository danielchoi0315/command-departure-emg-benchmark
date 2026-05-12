#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-https://dataverse.harvard.edu}"
DATA_ROOT="${DATA_ROOT:-$PWD/lake}"
OUT="${DATA_ROOT}/raw/ninapro_db10_meganepro"
BUNDLES="${OUT}/dataverse_bundles"
EXTRACT="${OUT}/dataverse_extracted"
STATUS_DIR="${OUT}/.status"
LOG_DIR="${OUT}/logs"
MIN_FILE_THRESHOLD="${MEGANEPRO_MIN_FILE_THRESHOLD:-40}"
BUNDLE_MAX_TIME="${MEGANEPRO_BUNDLE_MAX_TIME:-0}"
DIRINDEX_TIMEOUT="${MEGANEPRO_DIRINDEX_TIMEOUT:-0}"
API_MAX_FILES="${MEGANEPRO_API_MAX_FILES:-0}"
REDOWNLOAD_INVALID_BUNDLE="${MEGANEPRO_REDOWNLOAD_INVALID_BUNDLE:-0}"
API_EXT_ALLOWLIST="${MEGANEPRO_API_EXT_ALLOWLIST:-.mat,.tab,.txt,.csv,.tsv,.xlsx,.xls,.parquet,.json,.sfv}"

PIDS=(
  "doi:10.7910/DVN/1Z3IOM"
  "doi:10.7910/DVN/78QFZH"
  "doi:10.7910/DVN/F9R33N"
  "doi:10.7910/DVN/EJJ91H"
)

mkdir -p "${BUNDLES}" "${EXTRACT}" "${STATUS_DIR}" "${LOG_DIR}"

ensure_cmd() {
  local c="$1"
  if ! command -v "${c}" >/dev/null 2>&1; then
    echo "[meganepro] missing required command: ${c}" >&2
    exit 2
  fi
}

ensure_cmd curl
ensure_cmd unzip
ensure_cmd wget
ensure_cmd file
ensure_cmd grep
ensure_cmd awk
ensure_cmd sed
ensure_cmd python3

pid_safe() {
  printf "%s" "$1" | tr '/:.' '___'
}

curl_fetch_file() {
  # usage: curl_fetch_file URL OUTFILE
  local url="$1"
  local out="$2"
  local tmp="${out}.part"
  rm -f "${tmp}"
  if [[ -n "${DATAVERSE_KEY:-}" ]]; then
    curl -L --fail --show-error --silent -H "X-Dataverse-key:${DATAVERSE_KEY}" -o "${tmp}" "${url}"
  else
    curl -L --fail --show-error --silent -o "${tmp}" "${url}"
  fi
  mv -f "${tmp}" "${out}"
}

curl_download_bundle_with_cd() {
  # usage: curl_download_bundle_with_cd URL DESTDIR
  local url="$1"
  local dstdir="$2"
  mkdir -p "${dstdir}"
  (
    cd "${dstdir}"
    local cmd=(curl -L --fail --show-error --silent -O -J)
    if [[ "${BUNDLE_MAX_TIME}" =~ ^[0-9]+$ ]] && [[ "${BUNDLE_MAX_TIME}" -gt 0 ]]; then
      cmd+=(--max-time "${BUNDLE_MAX_TIME}")
    fi
    cmd+=(--retry 3 --retry-delay 2 --retry-connrefused)
    if [[ -n "${DATAVERSE_KEY:-}" ]]; then
      cmd+=(-H "X-Dataverse-key:${DATAVERSE_KEY}")
      cmd+=("${url}")
      "${cmd[@]}"
    else
      cmd+=("${url}")
      "${cmd[@]}"
    fi
  )
}

find_latest_file() {
  local d="$1"
  python3 - "$d" <<'PY'
from pathlib import Path
import sys
d = Path(sys.argv[1])
files = [p for p in d.iterdir() if p.is_file()]
if not files:
    print("")
    raise SystemExit(0)
files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(files[0].name)
PY
}

is_valid_zip() {
  local z="$1"
  [[ -s "${z}" ]] || return 1
  file "${z}" | grep -qi "Zip archive data" || return 1
  unzip -tqq "${z}" >/dev/null 2>&1 || return 1
  return 0
}

fetch_dataset_json() {
  local pid="$1"
  local out_json="$2"
  local url="${SERVER}/api/datasets/:persistentId/?persistentId=${pid}"
  curl_fetch_file "${url}" "${out_json}"
}

summarize_metadata() {
  # usage: summarize_metadata metadata.json summary.json files.tsv
  python3 - "$1" "$2" "$3" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

meta_path = Path(sys.argv[1])
summary_out = Path(sys.argv[2])
tsv_out = Path(sys.argv[3])

obj = json.loads(meta_path.read_text(encoding="utf-8"))
if obj.get("status") != "OK":
    raise SystemExit("metadata status is not OK")
files = obj.get("data", {}).get("latestVersion", {}).get("files", [])

rows = []
expected = 0
expected_public = 0
for f in files:
    df = f.get("dataFile", {}) or {}
    fid = df.get("id")
    fn = df.get("filename")
    if not fid or not fn:
        continue
    expected += 1
    restricted = bool(df.get("restricted", False))
    if not restricted:
        expected_public += 1
    dlabel = f.get("directoryLabel") or ""
    rel = f"{dlabel}/{fn}" if dlabel else str(fn)
    rows.append((int(fid), rel.replace("\\", "/"), int(restricted)))

summary = {
    "expected_file_count": expected,
    "expected_public_file_count": expected_public,
}
summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
with tsv_out.open("w", encoding="utf-8") as fh:
    for fid, rel, restricted in rows:
        fh.write(f"{fid}\t{rel}\t{restricted}\n")
PY
}

detect_manifest_omission() {
  local dir="$1"
  local manifest
  manifest="$(find "${dir}" -type f \( -iname "MANIFEST.TXT" -o -iname "manifest.txt" \) | head -n 1 || true)"
  if [[ -z "${manifest}" ]]; then
    echo "0"
    return 0
  fi
  if grep -Eiq "omitt|excluded|not included|restricted|zip size|too large|access denied|could not include" "${manifest}"; then
    echo "1"
  else
    echo "0"
  fi
}

count_files() {
  local d="$1"
  if [[ ! -d "${d}" ]]; then
    echo "0"
    return 0
  fi
  find "${d}" -type f | wc -l | awk '{print $1}'
}

count_key_files() {
  local d="$1"
  if [[ ! -d "${d}" ]]; then
    echo "0"
    return 0
  fi
  find "${d}" -type f \( -iname "*.mat" -o -iname "*.csv" -o -iname "*.tsv" -o -iname "*.txt" -o -iname "*.xlsx" -o -iname "*.xls" -o -iname "*.parquet" \) | wc -l | awk '{print $1}'
}

present_vs_expected() {
  # usage: present_vs_expected files.tsv dir_a dir_b dir_c out.json
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

tsv = Path(sys.argv[1])
dirs = [Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])]
out = Path(sys.argv[5])

def exists_any(rel: str) -> bool:
    for d in dirs:
        p = d / rel
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return True
    return False

present_public = 0
missing_public = 0
missing_restricted = 0

for line in tsv.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    parts = line.split("\t")
    if len(parts) != 3:
        continue
    _, rel, restricted = parts
    is_restricted = str(restricted).strip() == "1"
    has_file = exists_any(rel)
    if has_file and not is_restricted:
        present_public += 1
    if (not has_file) and is_restricted:
        missing_restricted += 1
    if (not has_file) and (not is_restricted):
        missing_public += 1

out.write_text(
    json.dumps(
        {
            "present_public_file_count": present_public,
            "missing_public_file_count": missing_public,
            "missing_restricted_file_count": missing_restricted,
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY
}

fallback_dirindex_wget() {
  # usage: fallback_dirindex_wget PID TARGET_DIR LOG_FILE
  local pid="$1"
  local target="$2"
  local log_file="$3"
  local url="${SERVER}/api/datasets/:persistentId/dirindex?persistentId=${pid}"
  mkdir -p "${target}"
  local cmd=(
    wget
    --recursive
    --no-parent
    --level=inf
    --no-host-directories
    --content-disposition
    --trust-server-names
    --directory-prefix="${target}"
    -e
    robots=off
  )
  if [[ "${DIRINDEX_TIMEOUT}" =~ ^[0-9]+$ ]] && [[ "${DIRINDEX_TIMEOUT}" -gt 0 ]]; then
    cmd+=(--timeout="${DIRINDEX_TIMEOUT}" --tries=1)
  fi
  if [[ -n "${DATAVERSE_KEY:-}" ]]; then
    cmd+=(--header="X-Dataverse-key:${DATAVERSE_KEY}")
  fi
  cmd+=("${url}")
  "${cmd[@]}" >"${log_file}" 2>&1 || return 1
  return 0
}

fallback_api_datafiles() {
  # usage: fallback_api_datafiles files.tsv dir_a dir_b api_dir log_file
  local tsv="$1"
  local dir_a="$2"
  local dir_b="$3"
  local api_dir="$4"
  local log_file="$5"
  mkdir -p "${api_dir}"
  python3 - "${tsv}" "${dir_a}" "${dir_b}" "${api_dir}" "${API_EXT_ALLOWLIST}" <<'PY' >"${log_file}.plan"
from __future__ import annotations
import sys
from pathlib import Path

tsv = Path(sys.argv[1])
dir_a = Path(sys.argv[2])
dir_b = Path(sys.argv[3])
api_dir = Path(sys.argv[4])
allow_raw = sys.argv[5] if len(sys.argv) > 5 else ""
allow_exts = {
    e.strip().lower()
    for e in (allow_raw or ".mat,.tab,.txt,.csv,.tsv,.xlsx,.xls,.parquet,.json,.sfv").split(",")
    if e.strip()
}

def exists_any(rel: str) -> bool:
    for d in (dir_a, dir_b, api_dir):
        p = d / rel
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return True
    return False

for line in tsv.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    parts = line.split("\t")
    if len(parts) != 3:
        continue
    fid, rel, restricted = parts
    if restricted.strip() == "1":
        continue
    rel = rel.replace("\\", "/")
    ext = Path(rel).suffix.lower()
    if allow_exts and ext and ext not in allow_exts:
        continue
    name = Path(rel).name.lower()
    if ext == ".mat" and "_orig" not in name:
        continue
    if exists_any(rel):
        continue
    print(f"{fid}\t{rel}")
PY

  local n_plan
  n_plan="$(wc -l < "${log_file}.plan" | awk '{print $1}')"
  echo "[meganepro] api fallback planned downloads: ${n_plan}" >"${log_file}"
  if [[ "${n_plan}" == "0" ]]; then
    return 0
  fi

  local n_done=0
  while IFS=$'\t' read -r fid rel; do
    if [[ "${API_MAX_FILES}" =~ ^[0-9]+$ ]] && [[ "${API_MAX_FILES}" -gt 0 ]] && [[ "${n_done}" -ge "${API_MAX_FILES}" ]]; then
      break
    fi
    [[ -z "${fid}" || -z "${rel}" ]] && continue
    local dest="${api_dir}/${rel}"
    mkdir -p "$(dirname "${dest}")"
    if [[ -s "${dest}" ]]; then
      continue
    fi
    local url="${SERVER}/api/access/datafile/${fid}?format=original"
    local tmp="${dest}.part"
    rm -f "${tmp}"
    if [[ -n "${DATAVERSE_KEY:-}" ]]; then
      curl -L --fail --show-error --silent -H "X-Dataverse-key:${DATAVERSE_KEY}" -o "${tmp}" "${url}" || continue
    else
      curl -L --fail --show-error --silent -o "${tmp}" "${url}" || continue
    fi
    if head -c 300 "${tmp}" | grep -Eiq "<!DOCTYPE|<html|\"status\"[[:space:]]*:[[:space:]]*\"ERROR\""; then
      rm -f "${tmp}"
      continue
    fi
    mv -f "${tmp}" "${dest}"
    n_done=$((n_done + 1))
  done < "${log_file}.plan"
  echo "[meganepro] api fallback downloaded files: ${n_done}" >>"${log_file}"
  return 0
}

write_pid_status() {
  # usage: write_pid_status status_json key value ...
  local out_json="$1"
  shift
  python3 - "$out_json" "$@" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
kv = sys.argv[2:]
obj = {}
for item in kv:
    if "=" not in item:
        continue
    k, v = item.split("=", 1)
    if v in {"true", "false"}:
        obj[k] = (v == "true")
        continue
    try:
        if "." in v:
            obj[k] = float(v)
        else:
            obj[k] = int(v)
        continue
    except Exception:
        pass
    obj[k] = v
out.write_text(json.dumps(obj, indent=2), encoding="utf-8")
PY
}

echo "[meganepro] SERVER=${SERVER}"
echo "[meganepro] OUT=${OUT}"

n_success=0
n_failed=0
n_incomplete=0

for PID in "${PIDS[@]}"; do
  SAFE="$(pid_safe "${PID}")"
  PID_BUNDLE_DIR="${BUNDLES}/${SAFE}"
  PID_EXTRACT_DIR="${EXTRACT}/${SAFE}"
  PID_DIRINDEX_DIR="${EXTRACT}/${SAFE}_dirindex"
  PID_API_DIR="${EXTRACT}/${SAFE}_api"
  PID_LOG_PREFIX="${LOG_DIR}/${SAFE}"
  PID_META_JSON="${OUT}/${SAFE}_metadata.json"
  PID_META_SUMMARY="${OUT}/${SAFE}_metadata_summary.json"
  PID_META_TSV="${OUT}/${SAFE}_metadata_files.tsv"
  PID_PRESENT_SUMMARY="${OUT}/${SAFE}_present_summary.json"
  PID_STATUS_JSON="${STATUS_DIR}/${SAFE}.json"

  mkdir -p "${PID_BUNDLE_DIR}" "${PID_EXTRACT_DIR}"
  echo "[meganepro] ===== PID ${PID} ====="

  bundle_url="${SERVER}/api/access/dataset/:persistentId/?persistentId=${PID}&format=original"

  # Step 1: dataset bundle download by persistentId.
  bundle_file=""
  existing_valid_zip="$(find "${PID_BUNDLE_DIR}" -maxdepth 1 -type f | head -n 1 || true)"
  if [[ -n "${existing_valid_zip}" ]] && is_valid_zip "${existing_valid_zip}"; then
    bundle_file="${existing_valid_zip}"
    echo "[meganepro] reusing existing valid bundle: ${bundle_file}"
  elif [[ -n "${existing_valid_zip}" ]] && [[ "${REDOWNLOAD_INVALID_BUNDLE}" != "1" ]]; then
    bundle_file="${existing_valid_zip}"
    echo "[meganepro] found existing invalid/partial bundle; skipping re-download and using fallbacks: ${bundle_file}"
  else
    before_latest="$(find_latest_file "${PID_BUNDLE_DIR}")"
    echo "[meganepro] downloading bundle via /api/access/dataset for ${PID}"
    if ! curl_download_bundle_with_cd "${bundle_url}" "${PID_BUNDLE_DIR}"; then
      echo "[meganepro] bundle download failed for ${PID}"
    fi
    after_latest="$(find_latest_file "${PID_BUNDLE_DIR}")"
    if [[ -n "${after_latest}" ]]; then
      bundle_file="${PID_BUNDLE_DIR}/${after_latest}"
    elif [[ -n "${before_latest}" ]]; then
      bundle_file="${PID_BUNDLE_DIR}/${before_latest}"
    fi
  fi

  zip_valid=0
  zip_name=""
  zip_size=0
  if [[ -n "${bundle_file}" ]] && [[ -f "${bundle_file}" ]] && is_valid_zip "${bundle_file}"; then
    zip_valid=1
    zip_name="$(basename "${bundle_file}")"
    zip_size="$(stat -c '%s' "${bundle_file}")"
    echo "[meganepro] bundle valid: ${zip_name} (${zip_size} bytes)"
  else
    echo "[meganepro] bundle invalid or missing for ${PID}"
  fi

  # Step 2: extract if valid zip.
  if [[ "${zip_valid}" == "1" ]]; then
    unzip -oq "${bundle_file}" -d "${PID_EXTRACT_DIR}"
  fi
  extracted_count="$(count_files "${PID_EXTRACT_DIR}")"
  manifest_present=0
  if find "${PID_EXTRACT_DIR}" -type f \( -iname "MANIFEST.TXT" -o -iname "manifest.txt" \) | head -n 1 | grep -q .; then
    manifest_present=1
  fi
  omission_detected="$(detect_manifest_omission "${PID_EXTRACT_DIR}")"
  key_file_count="$(count_key_files "${PID_EXTRACT_DIR}")"

  # Step 3: metadata + thresholds.
  metadata_ok=0
  expected_file_count=0
  expected_public_file_count=0
  if fetch_dataset_json "${PID}" "${PID_META_JSON}" && summarize_metadata "${PID_META_JSON}" "${PID_META_SUMMARY}" "${PID_META_TSV}"; then
    metadata_ok=1
    expected_file_count="$(python3 - <<PY
import json
obj=json.load(open("${PID_META_SUMMARY}", "r", encoding="utf-8"))
print(int(obj.get("expected_file_count", 0)))
PY
)"
    expected_public_file_count="$(python3 - <<PY
import json
obj=json.load(open("${PID_META_SUMMARY}", "r", encoding="utf-8"))
print(int(obj.get("expected_public_file_count", 0)))
PY
)"
  fi

  threshold="${MIN_FILE_THRESHOLD}"
  if [[ "${expected_public_file_count}" -gt 0 ]] && [[ "${expected_public_file_count}" -lt "${threshold}" ]]; then
    threshold="${expected_public_file_count}"
  fi

  need_fallback=0
  if [[ "${omission_detected}" == "1" ]]; then
    need_fallback=1
  fi
  if [[ "${extracted_count}" -lt "${threshold}" ]]; then
    need_fallback=1
  fi

  fallback_used="none"
  dirindex_count=0
  api_count=0

  # Fallback 1: dirindex + wget recursive.
  if [[ "${need_fallback}" == "1" ]]; then
    echo "[meganepro] triggering fallback 1 (dirindex+wget) for ${PID}"
    if fallback_dirindex_wget "${PID}" "${PID_DIRINDEX_DIR}" "${PID_LOG_PREFIX}_dirindex.log"; then
      fallback_used="dirindex"
    else
      echo "[meganepro] dirindex fallback failed for ${PID}"
    fi
  fi
  dirindex_count="$(count_files "${PID_DIRINDEX_DIR}")"

  # Fallback 2: metadata ids + per-file download for missing public files.
  if [[ "${metadata_ok}" == "1" ]]; then
    echo "[meganepro] running fallback 2 (per-file API) for ${PID}"
    if fallback_api_datafiles "${PID_META_TSV}" "${PID_EXTRACT_DIR}" "${PID_DIRINDEX_DIR}" "${PID_API_DIR}" "${PID_LOG_PREFIX}_api.log"; then
      if [[ "${fallback_used}" == "none" ]]; then
        fallback_used="api"
      elif [[ "${fallback_used}" == "dirindex" ]]; then
        fallback_used="dirindex+api"
      fi
    fi
  fi
  api_count="$(count_files "${PID_API_DIR}")"

  if [[ "${metadata_ok}" == "1" ]]; then
    present_vs_expected "${PID_META_TSV}" "${PID_EXTRACT_DIR}" "${PID_DIRINDEX_DIR}" "${PID_API_DIR}" "${PID_PRESENT_SUMMARY}"
  else
    cat >"${PID_PRESENT_SUMMARY}" <<JSON
{
  "present_public_file_count": 0,
  "missing_public_file_count": 999999,
  "missing_restricted_file_count": 0
}
JSON
  fi

  present_public="$(python3 - <<PY
import json
print(int(json.load(open("${PID_PRESENT_SUMMARY}", "r", encoding="utf-8")).get("present_public_file_count", 0)))
PY
)"
  missing_public="$(python3 - <<PY
import json
print(int(json.load(open("${PID_PRESENT_SUMMARY}", "r", encoding="utf-8")).get("missing_public_file_count", 0)))
PY
)"
  missing_restricted="$(python3 - <<PY
import json
print(int(json.load(open("${PID_PRESENT_SUMMARY}", "r", encoding="utf-8")).get("missing_restricted_file_count", 0)))
PY
)"

  total_count_pid="$(count_files "${PID_EXTRACT_DIR}")"
  total_count_pid=$(( total_count_pid + dirindex_count + api_count ))
  key_file_count_all="$(count_key_files "${PID_EXTRACT_DIR}")"
  key_file_count_all=$(( key_file_count_all + $(count_key_files "${PID_DIRINDEX_DIR}") + $(count_key_files "${PID_API_DIR}") ))

  complete=false
  status="incomplete"
  reason="incomplete_after_fallback"
  if [[ "${zip_valid}" == "1" || "${dirindex_count}" -gt 0 || "${api_count}" -gt 0 ]]; then
    if [[ "${metadata_ok}" == "1" ]]; then
      if [[ "${missing_public}" -eq 0 ]] && [[ "${key_file_count_all}" -gt 0 ]]; then
        complete=true
        status="complete"
        reason="all_public_files_present"
      fi
    else
      if [[ "${total_count_pid}" -ge "${threshold}" ]] && [[ "${key_file_count_all}" -gt 0 ]]; then
        complete=true
        status="complete"
        reason="threshold_and_keyfiles_met_without_metadata"
      fi
    fi
  else
    status="failed"
    reason="no_bundle_or_fallback_files"
  fi

  if [[ "${status}" == "complete" ]]; then
    n_success=$((n_success + 1))
  elif [[ "${status}" == "failed" ]]; then
    n_failed=$((n_failed + 1))
  else
    n_incomplete=$((n_incomplete + 1))
  fi

  write_pid_status "${PID_STATUS_JSON}" \
    "pid=${PID}" \
    "status=${status}" \
    "complete=${complete}" \
    "reason=${reason}" \
    "zip_valid=${zip_valid}" \
    "zip_file=${zip_name}" \
    "zip_size_bytes=${zip_size}" \
    "manifest_present=${manifest_present}" \
    "omission_detected=${omission_detected}" \
    "extracted_file_count=${extracted_count}" \
    "dirindex_file_count=${dirindex_count}" \
    "api_file_count=${api_count}" \
    "fallback_used=${fallback_used}" \
    "metadata_ok=${metadata_ok}" \
    "expected_file_count=${expected_file_count}" \
    "expected_public_file_count=${expected_public_file_count}" \
    "present_public_file_count=${present_public}" \
    "missing_public_file_count=${missing_public}" \
    "missing_restricted_file_count=${missing_restricted}" \
    "key_file_count=${key_file_count_all}" \
    "threshold=${threshold}"

  echo "[meganepro] PID=${PID} status=${status} complete=${complete} missing_public=${missing_public} fallback=${fallback_used}"
done

# Global adapter-oriented completeness signal.
adapter_available="$(PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" DATA_ROOT="${DATA_ROOT}" python3 - <<'PY'
from pathlib import Path
import os
from command_departure_benchmark.adapters import ninapro_db10_meganepro as m
root = Path(os.environ["DATA_ROOT"]) / "raw" / "ninapro_db10_meganepro"
print("1" if m.available(root) else "0")
PY
)"

python3 - "${OUT}" "${adapter_available}" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
adapter_available = bool(int(sys.argv[2]))
status_dir = out_root / ".status"
items = []
for p in sorted(status_dir.glob("*.json")):
    try:
        items.append(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        continue

summary = {
    "server": "https://dataverse.harvard.edu",
    "out_root": str(out_root),
    "adapter_available": adapter_available,
    "items": items,
}
(out_root / "inventory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[meganepro] wrote inventory: {out_root / 'inventory.json'}")
PY

echo "[meganepro] summary: success=${n_success} incomplete=${n_incomplete} failed=${n_failed}"
if [[ "${n_success}" -eq 0 && "${n_incomplete}" -eq 0 ]]; then
  echo "[meganepro] ERROR: all PIDs failed" >&2
  exit 1
fi
if [[ "${n_incomplete}" -gt 0 || "${n_failed}" -gt 0 ]]; then
  echo "[meganepro] INCOMPLETE: one or more PIDs are incomplete/failed; inspect ${OUT}/inventory.json" >&2
fi
exit 0
