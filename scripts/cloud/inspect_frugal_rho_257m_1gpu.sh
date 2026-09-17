#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs2/dimativator/frugal-rho-257m-1xc-wsd-20260915}
METRIC_TAIL_LINES=${METRIC_TAIL_LINES:-20}
echo "INSPECT_ROOT=${RESULTS_DIR}"

if [[ ! -d "${RESULTS_DIR}" ]]; then
    echo "RESULTS_ROOT_MISSING"
    exit 0
fi

while IFS= read -r metrics_file; do
    echo "METRICS_FILE=${metrics_file}"
    tail -n "${METRIC_TAIL_LINES}" "${metrics_file}" | sed 's/^/METRIC_JSON=/'
done < <(find "${RESULTS_DIR}" -type f -name metrics.jsonl -print | sort)

echo "MARKERS"
find "${RESULTS_DIR}" -maxdepth 1 -type f -name '.llama257M_frugal_rho*.done' -print | sort

echo "CHECKPOINTS"
find "${RESULTS_DIR}" -type f -path '*/ckpts/*/*.pt' \
    -printf '%s %T@ %p\n' | sort || true
