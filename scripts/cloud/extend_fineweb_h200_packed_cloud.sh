#!/usr/bin/env bash
set -euo pipefail

PACKED_DESTINATION=${PACKED_DESTINATION:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
PACKED_RANK=${PACKED_RANK:?PACKED_RANK must be 0 or 1}

python scripts/cloud/extend_fineweb_h200_packed.py \
    --manifest scripts/cloud/fineweb_h200_manifest.json.gz.b64 \
    --destination "${PACKED_DESTINATION}" \
    --rank "${PACKED_RANK}" \
    --initial-state "scripts/cloud/fineweb_h200_rank${PACKED_RANK}_after_1xc.json.gz"
