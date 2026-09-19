#!/usr/bin/env bash
set -euo pipefail

readonly secret_dir=/workspace-SR006.nfs3/dimativator/.secure_relay/slimadam_decay_20260919
if [[ -L "${secret_dir}" ]]; then
    echo "Refusing to follow secret directory symlink" >&2
    exit 2
fi
rm -f "${secret_dir}/private.pem" "${secret_dir}/public.pem"
rmdir "${secret_dir}"
echo SLIMADAM_DECAY_TRANSFER_KEY_REMOVED
