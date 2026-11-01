"""Read-only exact disk, campaign metrics, and bounded W&B connectivity probe."""

import base64
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

root = Path(os.environ['RESULTS_DIR'])
logs = '/home/jovyan/shares/SR006.nfs3/mlsub-logs'
paths = ('/workspace-SR006.nfs3', logs, '/home/jovyan', '/tmp', '/')
print('SPACE_SNAPSHOT=' + json.dumps(dict(timestamp=time.time(), logs_dir=logs,
      available_bytes={p: shutil.disk_usage(p).free for p in paths})), flush=True)
for path in sorted(root.glob('**/finals/*.json')):
    print('FINAL=' + json.dumps(json.loads(path.read_text())), flush=True)
for path in sorted(root.glob('**/disk_status.json')):
    print('WORKER_DISK=' + json.dumps(json.loads(path.read_text())), flush=True)
for path in sorted(root.glob('**/metrics.jsonl')):
    latest = {}
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[record['event']] = record
    print('PROGRESS=' + json.dumps(dict(path=str(path), **latest)), flush=True)
if os.environ.get('CHECK_WANDB') == '1':
    # Never print the request or response body: both may contain account data.
    key = os.environ.get('WANDB_API_KEY', '')
    auth = base64.b64encode(('api:' + key).encode()).decode()
    request = urllib.request.Request('https://wandb-radfan.ru/graphql',
              data=json.dumps({'query': '{ viewer { username } }'}).encode(),
              headers={'Content-Type': 'application/json', 'Authorization': 'Basic ' + auth})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            value = json.load(response)
        ready = bool((value.get('data') or {}).get('viewer')) and not value.get('errors')
        print('WANDB_READY=' + json.dumps(dict(ready=ready)), flush=True)
    except (urllib.error.URLError, TimeoutError, ValueError):
        print('WANDB_READY=' + json.dumps(dict(ready=False)), flush=True)
