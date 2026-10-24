"""Opt-in Cloud.ru storage reserve with coordinated DDP stop and cleanup."""

import json
import os
from pathlib import Path
import shutil
import time

import torch.distributed as dist


RESERVE_BYTES = 5_000_000_000


class DiskSpaceEmergency(RuntimeError):
    pass


def available(paths: list[Path]) -> list[dict]:
    return [dict(path=str(path), available_bytes=shutil.disk_usage(path).free,
                 device=path.stat().st_dev) for path in paths]


def cleanup_checkpoints(root: Path, paths: list[Path], extra_bytes: int = 0) -> list[str]:
    """Only numeric checkpoint directories in this campaign, without symlinks."""
    root = root.resolve(strict=True)
    candidates = []
    for path in root.glob('*/**/ckpts/*'):
        if not path.name.isdigit() or not path.is_dir() or not (path / 'main.pt').is_file():
            continue
        chain = [path, *path.relative_to(root).parents]
        if path.is_symlink() or any((root / part).is_symlink() for part in chain[1:]):
            continue
        if any(p.is_symlink() for p in path.rglob('*')):
            continue
        candidates.append(path)
    deleted = []
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime):
        deficient = {record['device'] for record in available(paths)
                     if record['available_bytes'] <= RESERVE_BYTES + (extra_bytes if record['device'] == root.stat().st_dev else 0)}
        if not deficient:
            break
        if path.stat().st_dev not in deficient:
            continue
        shutil.rmtree(path)
        deleted.append(str(path))
    return deleted


class DiskGuard:
    def __init__(self, root: Path, paths: list[Path]):
        self.root = root.resolve(strict=True)
        self.paths = paths

    def check(self, extra_bytes: int = 0) -> None:
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        try:
            local = dict(timestamp=time.time(), paths=available(self.paths), error=None, write_device=self.root.stat().st_dev)
        except OSError as exc:
            local = dict(timestamp=time.time(), paths=[], error=str(exc))
        records = [local]
        if distributed:
            records = [None] * dist.get_world_size()
            dist.all_gather_object(records, local)
        emergency = any(r['error'] or any(p['available_bytes'] <= RESERVE_BYTES + (extra_bytes if p['device'] == r['write_device'] else 0)
                                        for p in r['paths']) for r in records)
        if rank == 0:
            payload = dict(event='disk_guard', timestamp=time.time(), records=records,
                           reserve_bytes=RESERVE_BYTES, pending_write_bytes=extra_bytes,
                           emergency=emergency)
            print('DISK_GUARD=' + json.dumps(payload), flush=True)
            if not emergency:
                try:
                    target = self.root / 'disk_status.json'
                    temporary = self.root / 'disk_status.tmp'
                    temporary.write_text(json.dumps(payload))
                    temporary.replace(target)
                except OSError as exc:
                    print('DISK_GUARD_STATUS_WRITE_FAILED=' + str(exc), flush=True)
        if not emergency:
            return
        # All ranks have reached this collective and have paused training/writes.
        # Nonmaster ranks wait below while master cleans, then all ranks exit.
        if rank == 0:
            print('КРИТИЧЕСКИ МАЛО МЕСТА ИЛИ НЕДОСТУПЕН ЗАМЕР НА CLOUD.RU. '
                  'ОБУЧЕНИЕ ОСТАНОВЛЕНО. ПРОВЕРЯЮ ЧЕКПОИНТЫ.', flush=True)
            try:
                deleted = cleanup_checkpoints(self.root, self.paths, extra_bytes)
                audit = dict(before=records, deleted_checkpoint_paths=deleted,
                             after=available(self.paths), timestamp=time.time())
                print('DISK_GUARD_CLEANUP=' + json.dumps(audit), flush=True)
                try:
                    (self.root / 'disk_emergency.json').write_text(json.dumps(audit))
                except OSError:
                    pass
            except OSError as exc:
                print('DISK_GUARD_CLEANUP_FAILED=' + str(exc), flush=True)
        if distributed:
            dist.barrier()
        raise DiskSpaceEmergency('Storage reserve unavailable; stopped without automatic resume')


def from_environment() -> DiskGuard | None:
    root = os.environ.get('CLOUD_DISK_GUARD_ROOT')
    if not root:
        return None
    paths = [Path(p) for p in json.loads(os.environ['CLOUD_DISK_GUARD_PATHS'])]
    return DiskGuard(Path(root), paths)
