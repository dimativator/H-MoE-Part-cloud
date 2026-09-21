#!/usr/bin/env bash
set -euo pipefail

readonly secret_dir=/workspace-SR006.nfs3/dimativator/.secure_relay/stage3_wandb_sync_andrey_20260921
if [[ -L "${secret_dir}" ]]; then
    echo "Refusing to follow secret directory symlink" >&2
    exit 2
fi
rm -f "${secret_dir}/private.pem" "${secret_dir}/public.pem"
rmdir "${secret_dir}"
echo STAGE3_WANDB_ANDREY_SYNC_KEY_REMOVED
