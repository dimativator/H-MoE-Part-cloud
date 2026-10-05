#!/usr/bin/env bash
set -euo pipefail
echo "INSPECT_DATE=$(date --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
python - <<'PY'
import json
from pathlib import Path
p = Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed/packed_metadata.json')
m = json.loads(p.read_text())
print('PACKED_METADATA', {k: m.get(k) for k in ['format','iterations','world_size','batch_size','manifest_fingerprint']})
PY
if [[ -n "${RESULTS_DIR:-}" && -d "$RESULTS_DIR/logs" ]]; then
    for log in "$RESULTS_DIR"/logs/*.log; do
        echo "LOG=$log"
        tail -n 12 "$log"
    done
fi
