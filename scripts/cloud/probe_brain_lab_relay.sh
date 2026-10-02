#!/usr/bin/env bash
set -euo pipefail

readonly host=proxy2.cod.phystech.edu
readonly port=10210
readonly trusted_key='[proxy2.cod.phystech.edu]:10210 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIHHLCvWHzxMi+m7XnSqdhq3qWPOmGt+88zaEcWPDTui'

echo "PROBE_START=$(date --iso-8601=seconds) HOST=$(hostname)"
command -v ssh
command -v sftp
command -v ssh-keyscan
readonly observed_key=$(ssh-keyscan -T 8 -p "${port}" -t ed25519 "${host}" 2>/dev/null | head -n 1)
if [[ "${observed_key}" != "${trusted_key}" ]]; then
    echo 'BRAIN_LAB_RELAY_HOST_KEY_MISMATCH_OR_UNREACHABLE' >&2
    exit 2
fi
echo BRAIN_LAB_RELAY_NETWORK_AND_HOST_KEY_OK
