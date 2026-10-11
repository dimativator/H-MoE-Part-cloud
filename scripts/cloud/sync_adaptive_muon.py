"""Sync only verified completed campaign runs, preserving original W&B IDs."""
import json
import math
import os
import signal
from pathlib import Path
import subprocess

import wandb

os.environ['WANDB_BASE_URL'] = 'https://wandb-radfan.ru'
root = Path('/workspace-SR006.nfs3/dimativator/adaptive-muon-20261011')
def network_timeout(signum, frame):
    print('SYNC_DEFERRED=W&B API exceeded 90 seconds; offline artifacts preserved', flush=True)
    raise SystemExit(75)


signal.signal(signal.SIGALRM, network_timeout)
print('SYNC_START=initializing W&B API with a 90-second deadline', flush=True)
signal.alarm(90)
api = wandb.Api(timeout=30)
signal.alarm(0)
for marker in sorted((root / 'finals').glob('*.json')):
    final = json.loads(marker.read_text())
    if final.get('smoke'):
        continue
    run_id = final['wandb_url'].rsplit('/', 1)[1]
    matches = [p for p in (root / 'wandb').rglob('offline-run-*') if p.name.endswith('-' + run_id)]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one offline run for {run_id}, got {len(matches)}')
    run_dir = matches[0]
    synced = (run_dir / f'run-{run_id}.wandb.synced').exists()
    if not synced:
        subprocess.run(['wandb', 'sync', '--entity', 'andrey', '--project', 'fp8-pretrain',
                        '--mark-synced', str(run_dir)], check=True, timeout=300)
    signal.alarm(90)
    api.flush()
    remote = api.run(f'andrey/fp8-pretrain/{run_id}')
    if not math.isclose(remote.summary.get('final-val/loss', math.nan), final['val_loss'], abs_tol=1e-12, rel_tol=0):
        raise RuntimeError(f'Remote final loss mismatch: {run_id}')
    if remote.config.get('opt') != final['optimizer'] or remote.config.get('lr') != float(final['lr']):
        raise RuntimeError(f'Remote config mismatch: {run_id}')
    signal.alarm(0)
    print('WANDB_VERIFIED=' + json.dumps({'id': run_id, 'url': remote.url, 'val_loss': final['val_loss']}), flush=True)
