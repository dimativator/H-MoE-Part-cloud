#!/usr/bin/env bash
set -euo pipefail

readonly log_dir=/workspace-SR006.nfs3/dimativator/logs/slimadam_257m_fp8_states_cloud
readonly results_dir=/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud/8xChinchilla_257M_fp8_states/257m_slim_adam_fp8_states_8xC_cloud_1gpu_h100

log_file=$(find "${log_dir}" -maxdepth 1 -type f \
    -name '257m_slim_adam_fp8_states_8xC_cloud_1gpu_h100_full_*.log' \
    -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
if [[ -z "${log_file}" ]]; then
    echo "FULL_LOG_MISSING"
    exit 0
fi

echo "LOG=${log_file}"
echo "LOG_SIZE=$(stat -c %s "${log_file}")"
grep -aE 'GPU_STATUS|NVIDIA H100|PARITY_CONFIG|Resuming Training|FINEWEB_VALIDATION|>Eval:|Train: Iter=|Traceback|Error|TRAIN_EXIT|RELAY_' \
    "${log_file}" | tail -n 40 || true

if [[ -d "${results_dir}/ckpts/latest" ]]; then
    find "${results_dir}/ckpts/latest" -maxdepth 1 -type f \
        -printf 'CHECKPOINT_FILE %f %s\n' | sort
fi
