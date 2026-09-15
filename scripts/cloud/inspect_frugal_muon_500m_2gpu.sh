#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/frugal-muon-500m-2gpu-20260912}
INSPECT_TAIL_LINES=${INSPECT_TAIL_LINES:-512}
echo "INSPECT_ROOT=${RESULTS_DIR}"

if ! [[ "${INSPECT_TAIL_LINES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "INSPECT_TAIL_LINES must be a positive integer" >&2
    exit 2
fi

if [[ ! -d "${RESULTS_DIR}" ]]; then
    echo "RESULTS_ROOT_MISSING"
    exit 0
fi

while IFS= read -r metrics_file; do
    echo "METRICS_FILE=${metrics_file}"
    tail -n "${INSPECT_TAIL_LINES}" "${metrics_file}" | sed 's/^/METRIC_JSON=/'
done < <(find "${RESULTS_DIR}" -type f -name metrics.jsonl -print | sort)

echo "MARKERS"
find "${RESULTS_DIR}" -maxdepth 1 -type f -name '.*.done' -print | sort

echo "LATEST_CHECKPOINTS"
find "${RESULTS_DIR}" -type f -path '*/ckpts/*/*.pt' \
    -printf '%s %T@ %p\n' | sort || true
