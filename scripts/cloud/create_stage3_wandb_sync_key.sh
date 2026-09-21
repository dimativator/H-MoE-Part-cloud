#!/usr/bin/env bash
set -euo pipefail

readonly secret_dir=/workspace-SR006.nfs3/dimativator/.secure_relay/stage3_wandb_sync_20260921
readonly private_key="${secret_dir}/private.pem"
readonly public_key="${secret_dir}/public.pem"

umask 077
mkdir -p "${secret_dir}"
if [[ -e "${private_key}" || -e "${public_key}" ]]; then
    echo "W&B sync key already exists; refusing to overwrite it" >&2
    exit 2
fi
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 \
    -out "${private_key}" 2>/dev/null
openssl pkey -in "${private_key}" -pubout -out "${public_key}"
echo WANDB_SYNC_PUBLIC_KEY_BEGIN
base64 -w 0 "${public_key}"
echo
echo WANDB_SYNC_PUBLIC_KEY_END
