#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python scripts/cloud/sign_ademamix_precision.py "$@"
status=$?
printf 'SIGN_ENTRY_EXIT=%s\n' "${status}"
exit 0
