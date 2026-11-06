"""Read-only startup proofs, exact disk telemetry, and data-order run progress."""

import json
import os
from pathlib import Path
import time

from run_data_order_500m import ROOT, LOGS, TRIALS, emit, latest_metrics


def main() -> None:
    allowed = Path('/workspace-SR006.nfs3/dimativator/data-order-500m-20261106')
    if ROOT != allowed or os.environ.get('VERIFIED_BOOTSTRAP_LOGS_DIR') != LOGS:
        raise RuntimeError('Unexpected campaign root or bootstrap log routing')
    available = {}
    for path in ('/workspace-SR006.nfs3', LOGS, '/home/jovyan', '/tmp', '/'):
        stat = os.statvfs(path)
        available[path] = stat.f_bavail * stat.f_frsize
    emit('SPACE_SNAPSHOT', dict(timestamp=time.time(), available_bytes=available,
                               logs_dir=LOGS, reserve_bytes=5_000_000_000))
    for path in ('/workspace-SR006.nfs2', '/home/jovyan/shares/SR006.nfs2'):
        try:
            stat = os.statvfs(path)
            result = dict(path=path, available_bytes=stat.f_bavail * stat.f_frsize)
        except OSError as exc:
            result = dict(path=path, error=str(exc))
        emit('NFS2_READ_ONLY', dict(timestamp=time.time(), used_for_writes=False, **result))
    for trial in TRIALS:
        emit('PROGRESS', dict(trial=trial, metrics=latest_metrics(ROOT / trial / 'metrics.jsonl')))
        for rank in range(2):
            for suffix in ('_smoke', ''):
                path = ROOT / 'logs' / f'{trial}{suffix}_rank{rank}.log'
                if not path.exists():
                    continue
                with path.open('rb') as stream:
                    start = stream.read(1024 * 1024).decode(errors='replace')
                    stream.seek(max(0, path.stat().st_size - 131072))
                    end = stream.read().decode(errors='replace')
                markers = ('FINEWEB_PACKED_STEP_PERMUTATION', 'FINEWEB_PERMUTED_BATCH_SHA256',
                           'FINEWEB_VALIDATION_SHA256')
                proofs = [line for line in start.splitlines() if any(marker in line for marker in markers)]
                errors = [line for line in end.replace('\r', '\n').splitlines()
                          if any(marker in line for marker in ('Traceback', 'OutOfMemoryError', 'RuntimeError:', 'DISK_GUARD_CLEANUP'))]
                emit('LOG_PROOF', dict(trial=trial, rank=rank, smoke=bool(suffix), path=str(path),
                                       bytes=path.stat().st_size, modified_at=path.stat().st_mtime,
                                       proofs=proofs, errors=errors[-12:]))
    for path in sorted(ROOT.glob('finals/*.json')):
        emit('FINAL', json.loads(path.read_text()))
    for path in sorted(ROOT.glob('launcher_disk_*.json')):
        emit('WORKER_DISK', json.loads(path.read_text()))
    emit('CHECKPOINT_AUDIT', dict(paths=[str(p) for p in ROOT.glob('**/*.pt')]))


if __name__ == '__main__':
    main()
