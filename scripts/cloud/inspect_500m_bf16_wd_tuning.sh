#!/usr/bin/env bash
set -euo pipefail
echo "INSPECT_UTC=$(date -u --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
python - <<'PY'
import json
from pathlib import Path
roots = {
    'A': Path('/home/jovyan/dimativator/500m-bf16-wd-tuning-20261006'),
    'B': Path('/workspace-SR006.nfs2/dimativator/500m-bf16-wd-tuning-20261006'),
}
for queue, optimizers in (('A', ('frugal', 'galore')), ('B', ('frugal_mm', 'apollo'))):
    root = roots[queue]
    for opt in optimizers:
        for wd in ('1e-2', '1e-3', '1e-4'):
            name = f'llama500M_{opt}_bf16_wd{wd}_1xC_warmup2000_2gpu_20261006'
            exp = root / '500M_BF16_WD_tuning_1xC_20261006' / name
            latest = last_eval = None
            metrics = exp / 'metrics.jsonl'
            if metrics.is_file():
                with metrics.open() as stream:
                    for line in stream:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        latest = row
                        if row.get('event') == 'validation':
                            last_eval = row
            print('WD_STATUS', json.dumps({'queue': queue, 'optimizer': opt, 'wd': wd,
                  'wandb_id': f'wd500-{opt}-{wd}-20261006', 'latest': latest, 'last_eval': last_eval,
                  'verified': (root / 'markers' / f'{name}.done').is_file(),
                  'failure_markers': [p.name for p in (root / 'markers').glob(f'{name}.failed.*')],
                  'checkpoint_files': [str(p) for p in (exp / 'ckpts').rglob('*.pt')]}), flush=True)
    for mode in ('smoke', 'full'):
        log = root / 'logs' / f'queue_{queue}_{mode}_rank0.log'
        if log.is_file():
            lines = log.read_text(errors='replace').splitlines()
            selected = [line for line in lines if any(marker in line for marker in (
                'WD_RUN_START', 'WD_TUNING_COMPLETE', 'WD_QUEUE_', 'WD_FINAL_VERIFIED',
                'Traceback', 'Error:', 'OutOfMemoryError', 'No space left'))]
            print('WD_QUEUE_LOG', queue, mode, json.dumps(selected[-25:]), flush=True)
PY
