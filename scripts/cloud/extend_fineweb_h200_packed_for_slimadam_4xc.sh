#!/usr/bin/env bash
set -euo pipefail

readonly destination=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed
readonly manifest=scripts/cloud/fineweb_h200_manifest.json.gz.b64

df -h "${destination}"
if [[ "${PACKED_FINALIZE:-0}" == 1 ]]; then
    python scripts/cloud/extend_fineweb_h200_packed_for_slimadam_4xc.py \
        --destination "${destination}" --finalize
else
    : "${PACKED_RANK:?PACKED_RANK must be 0 or 1}"
    python scripts/cloud/extend_fineweb_h200_packed_for_slimadam_4xc.py \
        --destination "${destination}" --manifest "${manifest}" \
        --rank "${PACKED_RANK}"
fi
