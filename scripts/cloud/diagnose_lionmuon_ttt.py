"""Export this campaign's failure evidence when Cloud job-log API is unavailable."""
import json
import os
from pathlib import Path
import re
import time

ROOT = Path('/workspace-SR006.nfs3/dimativator/lionmuon-time-target-500m-20261115')
JOB = 'lm-mpi-job-d0f98b8a-6256-40fa-b0f7-3cd882b48aaf'
LOGS = Path('/home/jovyan/shares/SR006.nfs3/mlsub-logs')


def tail(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 16000))
        value = stream.read().decode(errors='replace')
    key = os.environ.get('WANDB_API_KEY', '')
    if key:
        value = value.replace(key, '[REDACTED]')
    return '\n'.join(line for line in value.splitlines() if not re.search(r'(?:api[_-]?key|password|secret)\s*[=:]', line, re.I))


def main() -> None:
    space = {}
    for path in (str(ROOT), str(LOGS), '/home/jovyan', '/tmp', '/'):
        stat = os.statvfs(path)
        space[path] = stat.f_bavail * stat.f_frsize
    payload = dict(timestamp=time.time(), job_id=JOB, available_bytes=space, logs={}, metrics={}, timing_plan=None)
    if any(v <= 5_100_000_000 for v in space.values()):
        print('КРИТИЧЕСКИ МАЛО МЕСТА НА CLOUD.RU: ' + json.dumps(space), flush=True)
        return
    for path in sorted((ROOT / 'logs').glob('bench_*_rank*.log')):
        payload['logs'][str(path)] = tail(path)
    bootstrap = LOGS / JOB
    if bootstrap.is_dir() and not bootstrap.is_symlink():
        for path in sorted(bootstrap.rglob('*'))[:20]:
            if path.is_file() and not path.is_symlink():
                payload['logs'][str(path)] = tail(path)
    for name in ('bench_muon_reference', 'bench_lion_muon'):
        path = ROOT / name / 'metrics.jsonl'
        payload['metrics'][name] = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
    plan = ROOT / 'timing_plan.json'
    if plan.is_file():
        payload['timing_plan'] = json.loads(plan.read_text())
    payload['checkpoint_paths'] = [str(p) for p in ROOT.glob('**/*.pt')]
    print('DIAGNOSTIC=' + json.dumps(payload, ensure_ascii=False), flush=True)
    import wandb
    run = wandb.init(entity='andrey', project='fp8-pretrain', id='lion-ttt-500m-diagnostic-20261115',
        name='LionMuon 500M paired timing failure diagnostics', resume='allow',
        dir=str(ROOT / 'wandb'), settings=wandb.Settings(init_timeout=60))
    run.summary['cloud_diagnostics'] = payload
    run.finish()
    print('DIAGNOSTIC_EXPORTED', flush=True)


if __name__ == '__main__':
    main()
