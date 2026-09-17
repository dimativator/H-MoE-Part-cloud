#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/frugal-muon-memory-cloud-20260917}
PRINT_ALL=${PRINT_ALL:-0}
EXPORT_BASE64=${EXPORT_BASE64:-0}
echo "RESULTS_DIR=${RESULTS_DIR}"
if [[ "${EXPORT_BASE64}" == "1" ]]; then
    echo "CSV_GZIP_BASE64_BEGIN"
    gzip -c "${RESULTS_DIR}/results.csv" | base64
    echo "CSV_GZIP_BASE64_END"
    exit 0
fi
if [[ -f "${RESULTS_DIR}/state.json" ]]; then
    echo "STATE=$(tr -d '\n' < "${RESULTS_DIR}/state.json")"
else
    echo "STATE_MISSING"
fi
if [[ -f "${RESULTS_DIR}/results.csv" ]]; then
    python - "${RESULTS_DIR}/results.csv" "${PRINT_ALL}" <<'PY'
import csv
import json
import sys

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
print(f"ROWS={len(rows)}")
selected = rows if sys.argv[2] == "1" else rows[-12:]
for row in selected:
    print("ROW=" + json.dumps(row, sort_keys=True))
PY
else
    echo "RESULTS_MISSING"
fi
if [[ -f "${RESULTS_DIR}/runner-full.log" ]]; then
    tail -n 20 "${RESULTS_DIR}/runner-full.log" | sed 's/^/RUNNER_TAIL=/'
fi
if [[ -d "${RESULTS_DIR}/logs" ]]; then
    latest_log=$(find "${RESULTS_DIR}/logs" -type f -name '*.log' -print0 \
        | xargs -0 ls -1t 2>/dev/null | head -n 1 || true)
    if [[ -n "${latest_log}" ]]; then
        echo "LATEST_CELL_LOG=${latest_log}"
        tail -n 120 "${latest_log}" | sed 's/^/CELL_TAIL=/'
    fi
fi
