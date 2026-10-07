#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python scripts/cloud/sign_ademamix_campaign.py "$@"
status=$?
printf 'SIGN_ENTRY_EXIT=%s\n' "${status}"
# Cloud.ru hides failed-job logs. The explicit exit marker preserves the failure.
exit 0
