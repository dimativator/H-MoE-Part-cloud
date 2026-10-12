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
    records = []
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print('METRIC_PARSE_ERROR', str(path), index, flush=True)
    train = [r for r in records if r.get('event') == 'train']
    val = [r for r in records if r.get('event') == 'validation']
    print('PROGRESS=' + json.dumps({'path': str(path), 'train': train[-1] if train else None,
                                   'validation': val[-1] if val else None}, sort_keys=True), flush=True)
for location in ('/home/jovyan', '/workspace-SR006.nfs2', '/workspace-SR006.nfs3'):
    print('DISK_FREE_GB', location, round(shutil.disk_usage(location).free / 1e9, 2), flush=True)

for path in sorted(root.glob('logs/*.log')):
    print('RUN_LOG', str(path), path.stat().st_size, flush=True)
    print('\n'.join(path.read_text().splitlines()[-15:]), flush=True)

if os.environ.get('INSPECT_CHECKPOINTS') == '1':
    import torch

    for directory in sorted(root.iterdir()):
        if directory.is_dir():
            size = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
            print('CAMPAIGN_BYTES', str(directory), size, flush=True)
    for checkpoint in sorted(root.glob('*/**/ckpts/67911')):
        info = {'path': str(checkpoint), 'files': {p.name: p.stat().st_size for p in checkpoint.iterdir() if p.is_file()}}
        try:
            main = torch.load(checkpoint / 'main.pt', map_location='cpu', weights_only=False, mmap=True)
            info['iteration'] = main['itr']
            info['model_tensors'] = len(main['model'])
            info['optimizer_states'] = len(main['optimizer']['state'])
            info['scheduler_present'] = main['scheduler'] is not None
            del main
            for rank in (0, 1):
                worker = torch.load(checkpoint / f'worker_{rank}.pt', map_location='cpu', weights_only=False)
                info[f'worker_{rank}_keys'] = sorted(worker)
                del worker
            info['loadable'] = True
        except Exception as exc:
            info['loadable'] = False
            info['error'] = f'{type(exc).__name__}: {exc}'
        print('CHECKPOINT=' + json.dumps(info, sort_keys=True), flush=True)
