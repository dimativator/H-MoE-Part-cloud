"""Read-only preflight and progress snapshots, never serialize checkpoints."""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('RESULTS_DIR', '/home/jovyan/dimativator/power-lion-muon-257m-lr-20261020'))
DATA = Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed')
GROUP = 'power_lion_muon_257m_lr_20261020'
EXPECTED = '7327154b810ec27cf5ca794aedcc3aea11796b218261ff24b5e2d3d2d283e00b'


def main():
    metadata = json.loads((DATA / 'packed_metadata.json').read_text())
    assert metadata['manifest_fingerprint'] == EXPECTED, 'Dataset fingerprint mismatch'
    print('DATA_FINGERPRINT_VERIFIED=' + EXPECTED, flush=True)
    print(f'ROOT={ROOT} EXISTS={ROOT.exists()}', flush=True)
    for mount in ('/home/jovyan', '/workspace-SR006.nfs3'):
        s = os.statvfs(mount)
        print(f'DISK path={mount} free_bytes={s.f_bavail * s.f_frsize}', flush=True)
    for lr in ('2e-3', '1e-3', '5e-4', '1e-4'):
        for mode in ('smoke', 'full'):
            name = f'power_lion_muon_257m_lr{lr}_wd1e-1_{mode}'
            path = ROOT / GROUP / name / 'metrics.jsonl'
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                for event in ('train', 'validation'):
                    subset = [r for r in rows if r.get('event') == event]
                    if subset:
                        print('METRIC_JSON=' + json.dumps(subset[-1]), flush=True)
                train = [r for r in rows if r.get('event') == 'train'][-11:]
                if len(train) >= 2:
                    seconds = (train[-1]['timestamp'] - train[0]['timestamp']) / (train[-1]['iter'] - train[0]['iter'])
                    remaining = max(0, 39250 - train[-1]['iter']) * seconds
                    print(f'ETA lr={lr} mode={mode} seconds_per_iter={seconds} remaining_seconds={remaining} caveat=extrapolation_last10percent_more_expensive', flush=True)
            for rank in (0, 1):
                log = ROOT / 'logs' / f'{name}_rank{rank}.log'
                if log.exists():
                    lines = log.read_text(errors='replace').splitlines()
                    for line in lines:
                        if any(key in line for key in ('POWER_LION_MUON_CONFIG', 'POWER_LION_MUON_GPU_TEST_OK', 'POWER_LION_MUON_COMPLETE', 'WANDB_DISABLED', 'Traceback', 'Error', 'CHECKPOINT')):
                            print(f'LOG lr={lr} mode={mode} rank={rank} {line}', flush=True)
                    print(f'LOG_TAIL lr={lr} mode={mode} rank={rank} ' + '\n'.join(lines[-3:]), flush=True)
    if ROOT.exists():
        checkpoints = list(ROOT.rglob('*.pt'))
        print('CHECKPOINT_FILES=' + json.dumps([str(p) for p in checkpoints]), flush=True)
        assert not checkpoints, 'Unexpected checkpoint file'


if __name__ == '__main__':
    main()
