#!/usr/bin/env bash
set -euo pipefail

exec /home/jovyan/hmoe-cloud/torch251-cu121/bin/python \
    scripts/cloud/delete_relayed_slimadam_checkpoints.py \
    --execute
