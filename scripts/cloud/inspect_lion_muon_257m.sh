#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import os, json
from pathlib import Path
root = Path(os.environ.get('RESULTS_DIR', '/workspace-SR006.nfs3/dimativator/lionmuon-probability-257m-20261012'))
for schedule in ['constant', 'linear_up', 'linear_down', 'decay_up', 'decay_down', 'periodic']:
    name = f'lionmuon_257m_{schedule}_full'
    print('SCHEDULE', schedule, flush=True)
    metrics = root / 'lionmuon_probability_257m_1xC_20261012' / name / 'metrics.jsonl'
    if metrics.exists():
        rows = [json.loads(l) for l in metrics.read_text().splitlines() if l.strip()]
        for event in ['train', 'validation']:
            subset = [r for r in rows if r.get('event') == event]
            if subset: print('METRIC_JSON=' + json.dumps(subset[-1]), flush=True)
    log = root / 'logs' / (name + '_rank0.log')
    if log.exists():
        for line in log.read_text().replace('\r', '\n').splitlines():
            if any(k in line for k in ['LION_MUON_CONFIG', 'wandb-radfan.ru/andrey/fp8-pretrain/runs/', 'Traceback', 'Error']): print(line, flush=True)
queue = root / 'logs' / 'queue_full_rank0.log'
if queue.exists():
    for line in queue.read_text().splitlines():
        if 'LION_MUON_COMPLETE' in line or 'LION_MUON_QUEUE_COMPLETE' in line: print(line, flush=True)
PY
df -h /home/jovyan /workspace-SR006.nfs3
