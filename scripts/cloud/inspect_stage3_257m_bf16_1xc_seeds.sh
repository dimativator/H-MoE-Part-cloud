#!/usr/bin/env bash
set -euo pipefail

readonly log_dir=/home/jovyan/dimativator/stage3_257m_bf16_1xc_seeds_20260923/logs

df -h /home/jovyan /workspace-SR006.nfs3
for optimizer in adamw muon; do
    for seed in 1 2 3; do
        log_file="${log_dir}/257m_${optimizer}_bf16_1xC_seed${seed}_cloud_1gpu.log"
        echo "RUN optimizer=${optimizer} seed=${seed}"
        if [[ ! -f "${log_file}" ]]; then
            echo LOG_NOT_YET_CREATED
            continue
        fi
        stat -c 'log_bytes=%s modified=%y' "${log_file}"
        grep -E 'SEED_RUN_START|Starting Experiment:|>Eval:|Iter=|SEED_RUN_COMPLETE|Traceback|Error|OOM|OutOfMemory' "${log_file}" | tail -n 8 || true
    done
done
