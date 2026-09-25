#!/usr/bin/env bash
set -euo pipefail

readonly root=/home/jovyan/dimativator/stage3_257m_scale_20260925
echo "INSPECT_DATE=$(date --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
for lr in 1e-4 5e-4 1e-3 2e-3; do
    log_file="${root}/logs/257m_scale_bf16_1xC_lr${lr}_cloud_2gpu_rank0.log"
    echo "SCALE_LR=${lr}"
    if [[ ! -f "${log_file}" ]]; then
        echo LOG_NOT_YET_CREATED
        continue
    fi
    stat -c 'LOG_BYTES=%s MODIFIED=%y' "${log_file}"
    tr '\r' '\n' < "${log_file}" | grep 'Train: Iter=' | tail -n 1 || true
    tr '\r' '\n' < "${log_file}" | grep '>Eval: Iter=' | tail -n 1 || true
    tr '\r' '\n' < "${log_file}" | \
        grep -E 'FINAL_VAL_LOSS_EXACT|SCALE_SWEEP_COMPLETE|Traceback|OutOfMemory|Error' | tail -n 4 || true
done

full_log="${root}/logs/257m_scale_bf16_4xC_cloud_4gpu_full_rank0.log"
if [[ -f "${full_log}" ]]; then
    echo SCALE_FULL_4XC
    tr '\r' '\n' < "${full_log}" | grep 'Train: Iter=' | tail -n 1 || true
    tr '\r' '\n' < "${full_log}" | grep '>Eval: Iter=' | tail -n 1 || true
    tr '\r' '\n' < "${full_log}" | \
        grep -E 'FINAL_VAL_LOSS_EXACT|SCALE_4XC_COMPLETE|Traceback|OutOfMemory|Error' | tail -n 4 || true
fi
for scale in 1 2; do
    decay_log="${root}/logs/257m_scale_bf16_${scale}xC_decay_cloud_4gpu_rank0.log"
    if [[ -f "${decay_log}" ]]; then
        echo "SCALE_DECAY=${scale}"
        tr '\r' '\n' < "${decay_log}" | grep 'Train: Iter=' | tail -n 1 || true
        tr '\r' '\n' < "${decay_log}" | grep '>Eval: Iter=' | tail -n 1 || true
        tr '\r' '\n' < "${decay_log}" | \
            grep -E 'FINAL_VAL_LOSS_EXACT|SCALE_DECAY_COMPLETE|Traceback|OutOfMemory|Error' | tail -n 4 || true
    fi
done
