"""Read-only storage, data, W&B, source, and optional CPU regressions."""

import os
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
if revision != os.environ['CODE_REVISION']:
    raise RuntimeError('Source revision mismatch')
if os.environ.get('ANVIL2_PREFLIGHT') == '1':
    env = {**os.environ, 'PYTHONPATH': str(repo / 'src') + ':' + str(repo), 'OMP_NUM_THREADS': '2'}
    env['CUDA_VISIBLE_DEVICES'] = ''
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                    '-p', 'test_anvil2.py', '-v'], cwd=repo, env=env, check=True)
    print('ANVIL2_REGRESSION_OK', flush=True)
import inspect_soft_lion_muon
