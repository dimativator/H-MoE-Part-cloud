"""Delete only the spectral dataset explicitly approved on 10 November 2026."""

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import time


ROOT = Path('/workspace-SR006.nfs3/dimativator')
TARGET = ROOT / 'spectral-wd-matched-tokenized'
LOGS = Path('/home/jovyan/shares/SR006.nfs3/mlsub-logs')
RECEIPT = ROOT / 'spectral_dataset_cleanup_20261110.json'
RESERVE = 5_000_000_000


def manifest(root: Path) -> list[dict]:
    """Reject symlinks and nested mounts before touching any file."""
    device = root.lstat().st_dev
    entries = []
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in folders + files:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or info.st_dev != device:
                raise RuntimeError(f'Refusing symlink or nested mount: {path}')
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise RuntimeError(f'Refusing special file: {path}')
            if stat.S_ISREG(info.st_mode):
                entries.append(dict(path=str(path.relative_to(root)), bytes=info.st_size))
    return entries


def available() -> dict[str, int]:
    result = {}
    for path in (ROOT, LOGS, Path('/home/jovyan'), Path('/tmp'), Path('/')):
        info = os.statvfs(path)
        result[str(path)] = info.f_bavail * info.f_frsize
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    before = available()
    if any(value <= RESERVE for value in before.values()):
        raise RuntimeError(f'Write filesystem reserve unavailable: {before}')
    if not LOGS.is_dir() or LOGS.stat().st_dev != ROOT.stat().st_dev:
        raise RuntimeError('NFS3 bootstrap log routing not verified')
    if TARGET.is_symlink() or not TARGET.is_dir():
        raise RuntimeError('Approved target is missing, a symlink, or not a directory')
    if TARGET.resolve().parent != ROOT.resolve():
        raise RuntimeError('Approved target is outside its recorded root')
    if TARGET.stat().st_dev != ROOT.stat().st_dev:
        raise RuntimeError('Approved target is on an unexpected filesystem')
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError('Safe directory removal unavailable')
    protected = [ROOT / 'fineweb-h200-packed', ROOT / 'iclr1b-dense-20261106']
    identities = {str(path): (path.stat().st_dev, path.stat().st_ino) for path in protected}
    target_identity = TARGET.lstat()
    files = manifest(TARGET)
    if not any(item['path'] == 'train.bin' for item in files):
        raise RuntimeError('Expected spectral train.bin missing')
    record = dict(timestamp=time.time(), target=str(TARGET), files=files,
                  apparent_bytes=sum(item['bytes'] for item in files),
                  available_before=before, apply=args.apply, status='planned')
    print('CLEANUP_PLAN=' + json.dumps(record, sort_keys=True), flush=True)
    if not args.apply:
        return
    current = TARGET.lstat()
    if (current.st_dev, current.st_ino) != (target_identity.st_dev, target_identity.st_ino):
        raise RuntimeError('Approved target changed during validation')
    shutil.rmtree(TARGET)
    if os.path.lexists(TARGET):
        raise RuntimeError('Approved target still exists after deletion')
    for path in protected:
        if (path.stat().st_dev, path.stat().st_ino) != identities[str(path)]:
            raise RuntimeError(f'Protected directory identity changed: {path}')
    after = available()
    record.update(status='deleted', completed_at=time.time(), available_after=after,
                  observed_nfs3_available_increase=after[str(ROOT)] - before[str(ROOT)],
                  protected_directory_identities_verified=True)
    temporary = RECEIPT.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(RECEIPT)
    print('CLEANUP_RESULT=' + json.dumps(record, sort_keys=True), flush=True)
    if any(value <= RESERVE for value in after.values()):
        raise RuntimeError('Filesystem below reserve after deletion; see receipt')


if __name__ == '__main__':
    main()
