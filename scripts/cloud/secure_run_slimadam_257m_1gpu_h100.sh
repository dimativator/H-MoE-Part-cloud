#!/usr/bin/env bash
set -euo pipefail

readonly branch=codex/slimadam-257m-1gpu-h100-resume
readonly encrypted_token_url="https://raw.githubusercontent.com/DimaTivator/H-MoE-Part-cloud/${branch}/scripts/cloud/.slimadam_relay_token.b64"
readonly secret_dir=/workspace-SR006.nfs3/dimativator/.secure_relay/slimadam_20260918
readonly private_key="${secret_dir}/private.pem"
readonly encrypted_token=/dev/shm/slimadam_hf_token.enc
readonly token_file=/dev/shm/slimadam_hf_token
readonly checkpoint_dir=/workspace-SR006.nfs3/dimativator/checkpoints/slimadam_257m_fp8_states_8xC/160000
readonly manifest_path=/home/jovyan/fineweb_h200_manifest.json

cleanup() {
    rm -f "${encrypted_token}" "${token_file}"
    if [[ "${DELETE_KEY_ON_EXIT:-0}" == "1" ]]; then
        rm -rf "${secret_dir}"
    fi
}
trap cleanup EXIT
umask 077

if [[ ! -s "${private_key}" ]]; then
    echo "Relay private key is missing" >&2
    exit 2
fi

for _ in $(seq 1 180); do
    if curl --fail --silent --show-error --location \
        "${encrypted_token_url}?nonce=$(date +%s)" \
        | base64 --decode > "${encrypted_token}" 2>/dev/null \
        && [[ -s "${encrypted_token}" ]] \
        && openssl pkeyutl -decrypt -inkey "${private_key}" \
            -pkeyopt rsa_padding_mode:oaep \
            -in "${encrypted_token}" -out "${token_file}" 2>/dev/null; then
        break
    fi
    rm -f "${encrypted_token}" "${token_file}"
    sleep 5
done
if [[ ! -s "${token_file}" ]]; then
    echo "Encrypted Hugging Face token was not received" >&2
    exit 3
fi

base64 --decode scripts/cloud/fineweb_h200_manifest.json.gz.b64 \
    | gzip --decompress > "${manifest_path}"

python scripts/cloud/download_hf_checkpoint.py \
    --repo-id DimaTivator/slimadam-257m-cloud-relay-20260915 \
    --remote-dir slimadam_257m_fp8_states_h200_resume/160000 \
    --destination "${checkpoint_dir}" \
    --token-file "${token_file}" \
    --expected-iteration 160000

export FINEWEB_MANIFEST="${manifest_path}"
export RESUME_FROM="${checkpoint_dir}"
export HF_TOKEN_FILE="${token_file}"
bash scripts/cloud/run_slimadam_257m_1gpu_h100_resume.sh
