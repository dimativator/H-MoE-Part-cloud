#!/usr/bin/env bash
set -euo pipefail
export GPU_COUNT=2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_slimadam_500m_precisions_4gpu.sh"
