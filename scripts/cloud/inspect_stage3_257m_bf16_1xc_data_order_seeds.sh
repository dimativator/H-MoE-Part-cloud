#!/usr/bin/env bash
set -euo pipefail

readonly log_dir=/home/jovyan/dimativator/stage3_257m_bf16_1xc_data_order_20260924/logs

df -h /home/jovyan /workspace-SR006.nfs3
for optimizer in adamw muon; do
    for seed in 1 2 3; do
        log_file="${log_dir}/257m_${optimizer}_bf16_1xC_data_order_seed${seed}_cloud_1gpu.log"
        echo "RUN optimizer=${optimizer} seed=${seed}"
        if [[ ! -f "${log_file}" ]]; then
            echo LOG_NOT_YET_CREATED
            continue
        fi
        stat -c 'log_bytes=%s modified=%y' "${log_file}"
        tr '\r' '\n' < "${log_file}" | grep 'FINEWEB_PACKED_STEP_PERMUTATION' | tail -n 1 || true
        tr '\r' '\n' < "${log_file}" | grep 'Train: Iter=' | tail -n 1 || true
        tr '\r' '\n' < "${log_file}" | grep '>Eval: Iter=' | tail -n 1 || true
        tr '\r' '\n' < "${log_file}" | \
            grep -E 'FINAL_VAL_LOSS_EXACT|DATA_ORDER_SEED_RUN_COMPLETE|Traceback|Error|OOM|OutOfMemory' | tail -n 3 || true
    done
done
