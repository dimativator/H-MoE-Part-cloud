#!/usr/bin/env bash
set -euo pipefail

readonly experiment_dir=/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud/8xChinchilla_257M_fp8_states/257m_slim_adam_fp8_states_8xC_cloud_4gpu
readonly encrypted_token_url=https://raw.githubusercontent.com/dimativator/H-MoE-Part-cloud/codex/slimadam-257m-4gpu-cloud/scripts/cloud/.slimadam_relay_token.b64
readonly secret_dir=/workspace-SR006.nfs3/dimativator/.secure_relay/slimadam_20260916
readonly private_key="${secret_dir}/private.pem"
readonly public_key="${secret_dir}/public.pem"
readonly encrypted_token="${secret_dir}/token.enc"
readonly token_file="${secret_dir}/token"
readonly stop_file="${secret_dir}/stop"

cleanup() {
    rm -rf "${secret_dir}"
}
trap cleanup EXIT
umask 077

if [[ ! -s "${private_key}" || ! -s "${public_key}" ]]; then
    echo "Relay keypair is missing" >&2
    exit 2
fi

for _ in $(seq 1 180); do
    if curl --fail --silent --show-error --location \
        "${encrypted_token_url}?nonce=$(date +%s)" \
        | base64 --decode > "${encrypted_token}" 2>/dev/null \
        && [[ -s "${encrypted_token}" ]]; then
        if openssl pkeyutl -decrypt -inkey "${private_key}" \
            -pkeyopt rsa_padding_mode:oaep \
            -in "${encrypted_token}" -out "${token_file}" 2>/dev/null; then
            break
        fi
    fi
    rm -f "${encrypted_token}" "${token_file}"
    sleep 5
done

if [[ ! -s "${token_file}" ]]; then
    echo "Encrypted Hugging Face token was not received" >&2
    exit 3
fi

touch "${stop_file}"
/home/jovyan/hmoe-cloud/torch251-cu121/bin/python \
    scripts/cloud/relay_latest_checkpoint_to_hf.py \
    --checkpoint-dir "${experiment_dir}/ckpts/latest" \
    --staging-dir "${experiment_dir}/ckpts/.relay" \
    --repo-id DimaTivator/slimadam-257m-cloud-relay-20260915 \
    --remote-prefix slimadam_257m_fp8_states_8xC \
    --world-size 4 \
    --stop-file "${stop_file}" \
    --token-file "${token_file}" \
    --poll-seconds 5
