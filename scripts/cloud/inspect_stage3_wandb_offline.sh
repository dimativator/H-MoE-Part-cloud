#!/usr/bin/env bash
set -euo pipefail

roots=(
    /workspace-SR006.nfs2/dimativator/500m-time-matched-20260925/adamw/wandb
    /home/jovyan/dimativator/500m-time-matched-20260925/slim_adam/wandb
    /workspace-SR006.nfs2/dimativator/500m-time-matched-20260925/frugal/wandb
    /home/jovyan/dimativator/stage3_257m_scale_20260925/wandb_offline
)

for root in "${roots[@]}"; do
    echo "WANDB_ROOT=${root}"
    if [[ ! -d "${root}" ]]; then
        echo "WANDB_ROOT_MISSING=${root}"
        continue
    fi
    while IFS= read -r run_dir; do
        run_id=${run_dir##*-}
        synced=false
        [[ -e "${run_dir}/.synced" ]] && synced=true
        echo "WANDB_OFFLINE_RUN id=${run_id} synced=${synced} path=${run_dir}"
        if [[ -f "${run_dir}/files/config.yaml" ]]; then
            grep -E '^(experiment_name|wandb_group|group|opt|lr):' "${run_dir}/files/config.yaml" | head -n 8 || true
        fi
        find "${run_dir}" -maxdepth 1 -type f -name '*.wandb' -printf 'WANDB_FILE=%f BYTES=%s\n'
    done < <(find "${root}" -maxdepth 3 -type d -name 'offline-run-*' -print | sort)
done

if [[ -n "${WANDB_API_KEY:-}" || -r "${HOME}/.netrc" || -r /home/jovyan/.netrc ]]; then
    echo 'WANDB_CREDENTIALS_PRESENT=true'
else
    echo 'WANDB_CREDENTIALS_PRESENT=false'
fi
