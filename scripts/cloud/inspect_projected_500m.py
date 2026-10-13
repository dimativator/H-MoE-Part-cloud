"""Read-only local metric and checkpoint snapshot for the projected campaign."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path('/home/jovyan/dimativator/projected-500m-precisions-20261013')
print('SNAPSHOT_UTC', datetime.now(timezone.utc).isoformat(), flush=True)
subprocess.run(['df', '-h', '/home/jovyan', '/workspace-SR006.nfs2', '/workspace-SR006.nfs3'], check=True)
subprocess.run(['df', '-i', '/home/jovyan'], check=True)
for metrics in root.glob('*/*/metrics.jsonl'):
    last_train = last_eval = latest = None
    finals = set()
    with metrics.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest = row
            if row.get('event') == 'train':
                last_train = row
            if row.get('event') == 'validation':
                last_eval = row
            if row.get('iter') in (75457, 150914):
                finals.add((row.get('iter'), row.get('event')))
    print('CAMPAIGN_METRICS', json.dumps({'path': str(metrics), 'last_train': last_train,
          'last_eval': last_eval, 'latest': latest, 'final_events': sorted(finals)}), flush=True)
for checkpoint in root.glob('*/*/ckpts/*/*.pt'):
    print('CHECKPOINT', str(checkpoint), checkpoint.stat().st_size, flush=True)
