#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs2/dimativator/galore-rho-257m-1xc-20260912}
echo "INSPECT_ROOT=${RESULTS_DIR}"

if [[ ! -d "${RESULTS_DIR}" ]]; then
    echo "RESULTS_ROOT_MISSING"
    exit 0
fi

while IFS= read -r metrics_file; do
    echo "METRICS_FILE=${metrics_file}"
    tail -n 10000 "${metrics_file}" | sed 's/^/METRIC_JSON=/'
done < <(find "${RESULTS_DIR}" -type f -name metrics.jsonl -print | sort)

echo "MARKERS"
find "${RESULTS_DIR}" -maxdepth 1 -type f -name '.llama257M_galore_rho*.done' -print | sort

echo "CHECKPOINTS"
find "${RESULTS_DIR}" -type f -path '*/ckpts/*/*.pt' \
    -printf '%s %T@ %p\n' | sort || true
