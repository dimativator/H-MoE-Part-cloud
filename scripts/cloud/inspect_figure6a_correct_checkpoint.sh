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
if [[ "${DIAGNOSE_RESUME:-0}" == 1 ]]; then
    python - <<'PY'
import os, torch
from pathlib import Path
root = Path(os.environ['RESULTS_DIR']) / 'figure6a_correct_checkpoint_20261005'
a = torch.load(root / 'smoke_continuous/ckpts/2/main.pt', map_location='cpu', mmap=True, weights_only=False)
b = torch.load(root / 'smoke_resume/ckpts/2/main.pt', map_location='cpu', mmap=True, weights_only=False)
max_model_diff = 0
for key, value in a['model'].items():
    other = b['model'][key]
    diff = (value.float() - other.float()).abs()
    maximum = diff.max().item()
    max_model_diff = max(max_model_diff, maximum)
    if maximum:
        print('MODEL_DIFF', key, 'max_abs', maximum, 'mean_abs', diff.mean().item(), 'mismatch', (diff > 0).sum().item())
print('MAX_MODEL_ABS_DIFF', max_model_diff)
for key in ['scheduler', 'param_groups']:
    left, right = (a[key], b[key]) if key == 'scheduler' else (a['optimizer'][key], b['optimizer'][key])
    print('EXACT_META', key, left == right)
for key in ['exp_avg','exp_avg_sq','scale_exp_avg','scale_exp_avg_sq']:
    maximum, count = 0, 0
    for index, state in a['optimizer']['state'].items():
        diff = (state[key].float() - b['optimizer']['state'][index][key].float()).abs()
        maximum = max(maximum, diff.max().item())
        count += (diff > 0).sum().item()
    print('OPT_DIFF', key, 'max_abs', maximum, 'mismatch', count)
PY
fi
