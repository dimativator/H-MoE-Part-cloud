"""Storage, packed data, regressions, and read-only campaign progress."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

repo = Path(__file__).resolve().parents[2]
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != os.environ['CODE_REVISION']:
    raise RuntimeError('Source revision mismatch')
for path in ('/workspace-SR006.nfs2', '/home/jovyan/shares/SR006.nfs2'):
    try:
        record = dict(path=path, available_bytes=shutil.disk_usage(path).free)
    except OSError as exc:
        record = dict(path=path, error=str(exc))
    print('NFS2_READ_ONLY=' + json.dumps(dict(timestamp=time.time(), **record)), flush=True)
if os.environ.get('SOFT_LION_MUON_PREFLIGHT') == '1':
    env = {**os.environ, 'PYTHONPATH': str(repo / 'src') + ':' + str(repo), 'OMP_NUM_THREADS': '2'}
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                    '-p', 'test_soft_lion_muon*.py', '-v'], check=True, env=env)
    print('SOFT_LION_MUON_REGRESSION_OK', flush=True)

import inspect_matched_normuon
