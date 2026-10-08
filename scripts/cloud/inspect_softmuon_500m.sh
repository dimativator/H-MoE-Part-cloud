#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path
root = Path('/workspace-SR006.nfs3/dimativator/softmuon-500m-20261008')
for precision in ['bf16', 'fp8_act', 'bf16_g16']:
    name = 'softmuon_500m_' + precision + '_full'
    path = root / 'softmuon_500m_1xC_20261008' / name / 'metrics.jsonl'
    print('PRECISION', precision, flush=True)
    if path.exists():
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for record in records[-3:]: print('METRIC_JSON=' + json.dumps(record), flush=True)
    log = root / 'logs' / (name + '_rank0.log')
    if log.exists():
        for line in log.read_text().replace('\r', '\n').splitlines():
            if any(key in line for key in ['SOFTMUON_CONFIG', 'wandb-radfan.ru/andrey/fp8-pretrain/runs/', 'SOFTMUON_COMPLETE', 'Traceback', 'Error']):
                print(line, flush=True)
PY
df -h /home/jovyan /workspace-SR006.nfs3
