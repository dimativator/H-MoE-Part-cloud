#!/usr/bin/env bash
set -euo pipefail

readonly secret_dir=/workspace-SR006.nfs3/dimativator/.secure_relay/stage3_wandb_sync_andrey_20260921
readonly private_key="${secret_dir}/private.pem"
readonly encrypted_key=scripts/cloud/stage3_wandb_sync_andrey_aes_key.b64
readonly encrypted_netrc=scripts/cloud/stage3_wandb_sync_andrey_netrc.b64
readonly wandb_root=/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud/wandb_offline/wandb
readonly -a run_ids=(33f6xomz tgfbbbo6 37hegcp5 0o7dtqv3)

tmp_dir=$(mktemp -d)
cleanup() {
    rm -f "${tmp_dir}/.netrc" "${tmp_dir}/aes.key" "${tmp_dir}/aes.key.enc"
    rmdir "${tmp_dir}"
}
trap cleanup EXIT

umask 077
base64 -d "${encrypted_key}" > "${tmp_dir}/aes.key.enc"
openssl pkeyutl -decrypt -inkey "${private_key}" \
    -in "${tmp_dir}/aes.key.enc" -out "${tmp_dir}/aes.key" \
    -pkeyopt rsa_padding_mode:oaep -pkeyopt rsa_oaep_md:sha256
openssl enc -d -aes-256-cbc -pbkdf2 -a -A \
    -in "${encrypted_netrc}" -out "${tmp_dir}/.netrc" \
    -pass "file:${tmp_dir}/aes.key"
chmod 600 "${tmp_dir}/.netrc"

export HOME="${tmp_dir}"
export WANDB_BASE_URL=https://wandb-radfan.ru
wandb login --verify

for run_id in "${run_ids[@]}"; do
    run_dir=$(find "${wandb_root}" -maxdepth 1 -type d \
        -name "offline-run-*-${run_id}" -print -quit)
    if [[ -z "${run_dir}" ]]; then
        echo "Missing offline run ${run_id}" >&2
        exit 3
    fi
    echo "WANDB_SYNC_START id=${run_id} path=${run_dir}"
    wandb sync --entity andrey --project fp8-pretrain \
        --include-offline --include-synced --mark-synced "${run_dir}"
    echo "WANDB_SYNC_DONE id=${run_id}"
done
echo STAGE3_SLIMADAM_WANDB_ANDREY_SYNC_COMPLETE
