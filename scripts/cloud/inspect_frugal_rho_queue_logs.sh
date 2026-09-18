#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs2/dimativator/frugal-rho-257m-1xc-wsd-20260915}
for queue in A B; do
    log_file="${RESULTS_DIR}/logs/queue_${queue}_full.log"
    echo "QUEUE_LOG=${log_file}"
    if [[ -f "${log_file}" ]]; then
        tail -n 80 "${log_file}"
    else
        echo "QUEUE_LOG_MISSING=${queue}"
    fi
done
