"""Read-only campaign snapshot for completed CPU inspector jobs."""
import json
import os
from pathlib import Path
import shutil

root = Path(os.environ.get('RESULTS_DIR', '/workspace-SR006.nfs3/dimativator/adaptive-muon-20261011'))
print('ROOT_EXISTS', str(root), root.exists(), flush=True)
for path in sorted(root.glob('finals/*.json')):
    print('FINAL=' + json.dumps(json.loads(path.read_text()), sort_keys=True), flush=True)
for path in sorted(root.glob('*/**/metrics.jsonl')):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    train = [r for r in records if r.get('event') == 'train']
    val = [r for r in records if r.get('event') == 'validation']
    print('PROGRESS=' + json.dumps({'path': str(path), 'train': train[-1] if train else None,
                                   'validation': val[-1] if val else None}, sort_keys=True), flush=True)
for location in ('/home/jovyan', '/workspace-SR006.nfs2', '/workspace-SR006.nfs3'):
    print('DISK_FREE_GB', location, round(shutil.disk_usage(location).free / 1e9, 2), flush=True)

for path in sorted(root.glob('logs/*.log')):
    print('RUN_LOG', str(path), path.stat().st_size, flush=True)
    print('\n'.join(path.read_text().splitlines()[-15:]), flush=True)
