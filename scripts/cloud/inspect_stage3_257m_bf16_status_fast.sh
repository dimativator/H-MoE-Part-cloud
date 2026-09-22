#!/usr/bin/env bash
set -euo pipefail

readonly root=/home/jovyan/dimativator/stage3_257m_bf16_native_20260921
readonly log_dir="${root}/logs"

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs3

for optimizer in frugal slim_adam; do
    echo "=== ${optimizer} ==="
    full_log="${log_dir}/257m_${optimizer}_bf16_native_states_8xC_cloud_4gpu_resume_rank0.log"
    if [[ -f "${full_log}" ]]; then
        echo "--- FULL ${full_log} ---"
        tail -c 2097152 "${full_log}" | tr '\r' '\n' | grep -E \
            'PHASE2_READER_RESUME_MODE|FULL_PHASE|FULL_8XC|>Eval:|Training Iteration|Traceback|OutOfMemory|NaN|No space left' \
            | tail -80 || true
    fi
    decay_log="${log_dir}/257m_${optimizer}_bf16_native_states_4xC_decay_cloud_1gpu.log"
    if [[ -f "${decay_log}" ]]; then
        echo "--- DECAY ${decay_log} ---"
        tail -c 1048576 "${decay_log}" | tr '\r' '\n' | grep -E \
            'DECAY_(START|COMPLETE)|QUEUE_COMPLETE|>Eval:|Training Iteration|Traceback|OutOfMemory|NaN|No space left' \
            | tail -60 || true
    fi
done
