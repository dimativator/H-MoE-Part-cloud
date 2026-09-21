#!/usr/bin/env bash
set -euo pipefail

readonly root=/home/jovyan/dimativator/stage3_257m_bf16_native_20260921
df -h /home/jovyan /workspace-SR006.nfs3
for optimizer in frugal slim_adam; do
    echo "=== ${optimizer} ==="
    find "${root}/logs" -maxdepth 1 -type f -name "*${optimizer}*" -print 2>/dev/null | sort
    while IFS= read -r log_file; do
        echo "--- ${log_file} ---"
        tr '\r' '\n' < "${log_file}" | grep -E \
            'FULL_PHASE|FULL_8XC|WAITING_FOR_CHECKPOINT|CHECKPOINT_READY|DECAY_(START|COMPLETE)|QUEUE_COMPLETE|>Eval:|Training Iteration|train/loss|Traceback|OutOfMemory|NaN|No space left' \
            | tail -40 || true
    done < <(find "${root}/logs" -maxdepth 1 -type f -name "*${optimizer}*" -print 2>/dev/null | sort)
    find "${root}" -path "*${optimizer}*/ckpts/*/main.pt" -o \
        -path "*${optimizer}*/ckpts/latest/main.pt" 2>/dev/null | sort || true
done
