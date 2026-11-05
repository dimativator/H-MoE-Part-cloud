"""Read-only storage, dataset, numerical regression, and campaign inspection."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

if os.environ.get('SOFTMUON_PREFLIGHT') == '1':
    for path in ('/workspace-SR006.nfs2', '/home/jovyan/shares/SR006.nfs2'):
        try:
            record = dict(path=path, available_bytes=shutil.disk_usage(path).free)
        except OSError as exc:
            record = dict(path=path, error=str(exc))
        print('NFS2_READ_ONLY=' + json.dumps(dict(timestamp=time.time(), **record)), flush=True)
    subprocess.run([sys.executable, 'scripts/cloud/test_softmuon_algorithm3.py'], check=True)

import inspect_matched_normuon
