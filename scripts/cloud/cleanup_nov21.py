import json
import os
import shutil
import subprocess
import time
from pathlib import Path

root = Path('/workspace-SR006.nfs3/dimativator')
targets = ('iclr500m-wd-precision-20261106', 'iclr1b-dense-20261106', 'data-order-500m-20261106')
paths = ('/', '/tmp', '/home/jovyan', '/workspace-SR006.nfs2', str(root))

def snapshot():
    values = {}
    for path in paths:
        stat = os.statvfs(path)
        values[path] = stat.f_bavail * stat.f_frsize
    print('SPACE', time.time(), json.dumps(values), flush=True)
    return values

before = snapshot()
if min(before.values()) <= 5_000_000_000:
    raise SystemExit('reserve unavailable')
if root.resolve(strict=True) != root or not shutil.rmtree.avoids_symlink_attacks:
    raise SystemExit('unsafe root or platform')
for name in targets:
    path = root / name
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise SystemExit(str(path))
    if path.is_dir():
        for base, dirs, _ in os.walk(path, followlinks=False):
            if os.path.ismount(base) or any(os.path.ismount(Path(base) / d) for d in dirs):
                raise SystemExit('nested mount')
for name in targets:
    path = root / name
    if path.exists():
        usage = subprocess.check_output(['du', '-sx', '-B1', '--', str(path)], text=True)
        print('REMOVING', usage.strip(), flush=True)
        shutil.rmtree(path)
    assert not os.path.lexists(path)
    print('ABSENT', path, flush=True)
snapshot()
print('FINEWEB_CONTENTS', [(p.name, p.lstat().st_size) for p in sorted((root / 'fineweb-h200-packed').iterdir())], flush=True)
for _ in range(4):
    time.sleep(10)
    if snapshot()[str(root)] > before[str(root)] + 1_000_000_000:
        break
