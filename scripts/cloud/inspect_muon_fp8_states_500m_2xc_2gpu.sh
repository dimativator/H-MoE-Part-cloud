#!/usr/bin/env bash
set -euo pipefail

results_dir=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/muon-fp8-states-500m-2xc-2gpu-20260918}
group=2xChinchilla_500M_muon_fp8_states_2gpu_cloud
experiment=llama500M_muon_fp8_states_bf16_wd1e-4_2xC_2gpu
experiment_dir="${results_dir}/${group}/${experiment}"
metrics_file="${experiment_dir}/metrics.jsonl"

echo "EXPERIMENT_DIR=${experiment_dir}"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 || true

if [[ -f "${metrics_file}" ]]; then
    echo "METRICS_FILE=${metrics_file}"
    tail -n 512 "${metrics_file}" | grep '"event": "train"' | tail -n 3 | sed 's/^/TRAIN=/' || true
    tail -n 512 "${metrics_file}" | grep '"event": "validation"' | tail -n 3 | sed 's/^/VALIDATION=/' || true
else
    echo "METRICS_MISSING=${metrics_file}"
fi

find "${results_dir}/logs" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -n 2 | while read -r _ log_file; do
        echo "LOG_FILE=${log_file}"
        tail -n 100 "${log_file}"
    done

find "${experiment_dir}/ckpts" -mindepth 2 -maxdepth 2 -type f \
    \( -name 'main.pt' -o -name 'worker_0.pt' -o -name 'worker_1.pt' \) \
    -printf 'CHECKPOINT=%s %T@ %p\n' 2>/dev/null | sort || true
