#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
scratch=$(mktemp -d)
trap 'rm -rf -- "${scratch}"' EXIT
mkdir -p "${scratch}/source"
worker_count=${BRAIN_LAB_CHECKPOINT_WORKERS:-4}
case "${worker_count}" in 2|4) ;; *) echo 'Invalid worker count' >&2; exit 2 ;; esac
files=(main.pt)
for (( rank=0; rank<worker_count; rank++ )); do files+=("worker_${rank}.pt"); done
for name in "${files[@]}"; do
    printf 'relay smoke %s\n' "${name}" > "${scratch}/source/${name}"
done
remote_name="relay-smoke-$(date +%s)-${RANDOM}"
python scripts/cloud/relay_brain_lab_checkpoint.py upload "${scratch}/source" "${remote_name}"
python scripts/cloud/relay_brain_lab_checkpoint.py download "${scratch}/download" "${remote_name}"
for name in "${files[@]}"; do
    cmp "${scratch}/source/${name}" "${scratch}/download/${name}"
done
echo "BRAIN_LAB_RELAY_ROUNDTRIP_OK name=${remote_name}"
