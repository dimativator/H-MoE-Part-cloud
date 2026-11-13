"""Read-only storage, regression preflight, and full validation-curve export."""

import json
import os
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
if revision != os.environ['CODE_REVISION']:
    raise RuntimeError('Unexpected code revision')
if os.environ.get('COSINE_PREFLIGHT') == '1':
    env = {**os.environ, 'PYTHONPATH': str(repo / 'src') + ':' + str(repo),
           'OMP_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': ''}
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                    '-p', 'test_cosine_257m.py', '-v'], cwd=repo, env=env, check=True)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                    '-p', 'test_soft_lion_muon.py', '-v'], cwd=repo, env=env, check=True)
    # Import vendored original references in the same image as training.
    subprocess.run([sys.executable, '-c',
        'from third_party.lion_muon.lion_muon import LionMuon; '
        'from third_party.lion_muon.muon import Muon; '
        'from third_party.softsign.softmuon import SingleDeviceSoftMuonWithAuxAdam'],
        cwd=repo, env=env, check=True)
    print('COSINE_REGRESSION_OK', flush=True)

import inspect_soft_lion_muon

root = Path(os.environ['RESULTS_DIR'])
for path in sorted(root.glob('**/metrics.jsonl')):
    if 'cosine_1xC' not in str(path):
        continue
    validation, train_tail = {}, []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('event') == 'validation':
            validation[row['iter']] = row
        elif row.get('event') == 'train':
            train_tail.append(row)
            train_tail = train_tail[-12:]
    # The log service drops oversized entries once a curve exceeds its limit.
    # Keep every entry below the MPI framing size and verify counts downstream.
    for row in validation.values():
        print('CURVE_VALIDATION=' + json.dumps(dict(path=str(path), record=row)), flush=True)
    for row in train_tail:
        print('CURVE_TRAIN=' + json.dumps(dict(path=str(path), record=row)), flush=True)
    print('CURVE_COMPLETE=' + json.dumps(dict(path=str(path),
          validation_points=len(validation), train_points=len(train_tail))), flush=True)
# Report paths only, never delete or traverse checkpoint symlinks.
checkpoints = [str(p) for p in root.glob('**/*.pt') if p.is_file() and not p.is_symlink()]
print('CHECKPOINT_AUDIT=' + json.dumps(dict(paths=checkpoints)), flush=True)
