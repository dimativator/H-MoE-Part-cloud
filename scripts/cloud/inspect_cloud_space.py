"""Read-only exact disk measurements for the NFS3 campaign preflight."""
import json
from pathlib import Path
import shutil
import time

logs = '/home/jovyan/shares/SR006.nfs3/mlsub-logs'
paths = ('/workspace-SR006.nfs3', logs, '/home/jovyan', '/tmp')
record = dict(timestamp=time.time(), logs_dir=logs,
              available_bytes={path: shutil.disk_usage(Path(path)).free for path in paths},
              source='completed_read_only_cpu_probe')
print('SPACE_SNAPSHOT=' + json.dumps(record, sort_keys=True), flush=True)
