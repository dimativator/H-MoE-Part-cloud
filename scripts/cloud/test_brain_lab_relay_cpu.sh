#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
scratch=$(mktemp -d)
trap 'rm -rf -- "${scratch}"' EXIT
mkdir -p "${scratch}/source"
for name in main.pt worker_0.pt worker_1.pt worker_2.pt worker_3.pt; do
    printf 'relay smoke %s\n' "${name}" > "${scratch}/source/${name}"
done
remote_name="relay-smoke-$(date +%s)-${RANDOM}"
python scripts/cloud/relay_brain_lab_checkpoint.py upload "${scratch}/source" "${remote_name}"
python scripts/cloud/relay_brain_lab_checkpoint.py download "${scratch}/download" "${remote_name}"
for name in main.pt worker_0.pt worker_1.pt worker_2.pt worker_3.pt; do
    cmp "${scratch}/source/${name}" "${scratch}/download/${name}"
done
echo "BRAIN_LAB_RELAY_ROUNDTRIP_OK name=${remote_name}"
