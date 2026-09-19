#!/usr/bin/env bash
set -euo pipefail

readonly secret_dir=/workspace-SR006.nfs3/dimativator/.secure_relay/slimadam_decay_20260919
readonly private_key="${secret_dir}/private.pem"
readonly encrypted_file=scripts/cloud/.slimadam_decay_token.b64
readonly checkpoint_root=/workspace-SR006.nfs3/dimativator/exps/8xChinchilla_257M_fp8_states_cloud/8xChinchilla_257M_fp8_states/257m_slim_adam_fp8_states_8xC_cloud_4gpu/ckpts
readonly repo_id=DimaTivator/slimadam-257m-cloud-relay-20260915
readonly remote_prefix=slimadam_257m_fp8_states_8xC

umask 077
if [[ ! -s "${private_key}" || ! -s "${encrypted_file}" ]]; then
    echo "Decay transfer key or encrypted token is missing" >&2
    exit 2
fi
encrypted_temp=$(mktemp /dev/shm/slimadam-decay-token.XXXXXX.enc)
token_file=$(mktemp /dev/shm/slimadam-decay-token.XXXXXX)
trap 'rm -f "${encrypted_temp}" "${token_file}"' EXIT
base64 --decode "${encrypted_file}" > "${encrypted_temp}"
openssl pkeyutl -decrypt -inkey "${private_key}" \
    -pkeyopt rsa_padding_mode:oaep \
    -in "${encrypted_temp}" -out "${token_file}"

for iteration in 35325 70650 141300; do
    python scripts/cloud/download_hf_checkpoint.py \
        --repo-id "${repo_id}" \
        --remote-dir "${remote_prefix}/${iteration}" \
        --destination "${checkpoint_root}/${iteration}" \
        --token-file "${token_file}" \
        --expected-iteration "${iteration}" \
        --expected-world-size 4
done
touch "${checkpoint_root}/slimadam_decay_sources_ready"
echo SLIMADAM_DECAY_SOURCES_READY
