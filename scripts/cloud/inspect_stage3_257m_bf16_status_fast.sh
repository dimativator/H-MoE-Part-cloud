#!/usr/bin/env bash
set -euo pipefail

readonly root=/home/jovyan/dimativator/stage3_257m_bf16_native_20260921
readonly log_dir="${root}/logs"
readonly full_suffix=${INSPECT_FULL_SUFFIX:-}
readonly decay_suffix=${INSPECT_DECAY_SUFFIX:-}
readonly full_mode=${INSPECT_FULL_MODE:-resume}
[[ "${full_suffix}" =~ ^[a-zA-Z0-9_-]*$ && "${decay_suffix}" =~ ^[a-zA-Z0-9_-]*$ ]] || {
    echo "Inspection suffixes must contain only letters, digits, underscores, or hyphens" >&2
    exit 2
}
[[ "${full_mode}" == full || "${full_mode}" == resume || "${full_mode}" == resume_packed ]] || {
    echo "INSPECT_FULL_MODE must be full, resume, or resume_packed" >&2
    exit 2
}

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs3

for optimizer in frugal slim_adam; do
    if [[ -n "${INSPECT_OPTIMIZER:-}" && "${optimizer}" != "${INSPECT_OPTIMIZER}" ]]; then
        continue
    fi
    echo "=== ${optimizer} ==="
    full_log="${log_dir}/257m_${optimizer}_bf16_native_states_8xC_cloud_4gpu${full_suffix}_${full_mode}_rank0.log"
    if [[ -f "${full_log}" ]]; then
        echo "--- FULL ${full_log} ---"
        tail -c 2097152 "${full_log}" | tr '\r' '\n' | grep -E \
            'PHASE2_READER_RESUME_MODE|FULL_PHASE|FULL_8XC|>Eval:|Training Iteration|Traceback|OutOfMemory|NaN|No space left' \
            | tail -80 || true
    fi
    decay_log="${log_dir}/257m_${optimizer}_bf16_native_states_4xC_decay_cloud_1gpu${decay_suffix}.log"
    if [[ -f "${decay_log}" ]]; then
        echo "--- DECAY ${decay_log} ---"
        if [[ "${INSPECT_RAW_DECAY:-0}" == 1 ]]; then
            stat -c 'DECAY_LOG_SIZE=%s DECAY_LOG_MTIME=%y' "${decay_log}"
            tail -n 80 "${decay_log}"
            continue
        fi
        tail -c 1048576 "${decay_log}" | tr '\r' '\n' | grep -E \
            'DECAY_(START|COMPLETE)|QUEUE_COMPLETE|>Eval:|Training Iteration|Traceback|OutOfMemory|NaN|No space left' \
            | tail -60 || true
    fi
done
