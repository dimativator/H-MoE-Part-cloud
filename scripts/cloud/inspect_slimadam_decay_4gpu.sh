#!/usr/bin/env bash
set -euo pipefail

readonly log_dir=/workspace-SR006.nfs3/dimativator/logs/slimadam_257m_fp8_states_cloud

for scale in 1 2 4; do
    for rank in 0 1 2 3; do
        log_file="${log_dir}/257m_slim_adam_fp8_states_${scale}xC_decay_cloud_4gpu_rank${rank}.log"
        echo "DECAY_LOG scale=${scale} rank=${rank} path=${log_file}"
        if [[ ! -f "${log_file}" ]]; then
            echo "DECAY_LOG_MISSING scale=${scale} rank=${rank}"
            continue
        fi
        echo "DECAY_LOG_BYTES=$(stat -c %s "${log_file}")"
        tr '\r' '\n' < "${log_file}" | grep -aE 'Train: Iter=|>Eval:|DECAY_EXIT|Traceback|Error|Exception|No space left|Killed|OutOfMemory|CUDA out of memory' | tail -n 12 || true
        if [[ "${scale}" == 4 ]]; then
            echo "DECAY_LOG_TAIL scale=4 rank=${rank}"
            tr '\r' '\n' < "${log_file}" | tail -n 55
        fi
    done
done
