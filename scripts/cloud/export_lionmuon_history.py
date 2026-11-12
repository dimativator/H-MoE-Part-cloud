"""Export six existing LionMuon histories to job stdout without training or writes."""
import base64
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import zlib

ROOT = Path('/home/jovyan/dimativator/lionmuon-probability-257m-nowandb-20261015')
GROUP = 'lionmuon_probability_257m_1xC_20261012'
SCHEDULES = ('constant', 'linear_up', 'linear_down', 'decay_up', 'decay_down', 'periodic')
PATHS = ('/home/jovyan', '/workspace-SR006.nfs3',
         '/home/jovyan/shares/SR006.nfs3/mlsub-logs', '/', '/tmp')


def guard() -> None:
    available = {p: os.statvfs(p).f_bavail * os.statvfs(p).f_frsize for p in PATHS}
    ready = all(v > 5_000_000_000 for v in available.values())
    print('EXPORT_SPACE=' + json.dumps(dict(timestamp=time.time(), available_bytes=available,
                                           ready=ready)), flush=True)
    if not ready:
        os._exit(72)


def watchdog(stop: threading.Event) -> None:
    while not stop.wait(30):
        guard()


def main() -> None:
    guard()
    for p in ('/workspace-SR006.nfs2', '/home/jovyan/shares/SR006.nfs2'):
        try:
            s = os.statvfs(p)
            available = s.f_bavail * s.f_frsize
        except OSError:
            available = None
        print('NFS2_READ_ONLY=' + json.dumps(dict(path=p, available_bytes=available,
                                                used_for_writes=False)), flush=True)
    stop = threading.Event()
    threading.Thread(target=watchdog, args=(stop,), daemon=True).start()
    try:
        for schedule in SCHEDULES:
            directory = ROOT / GROUP / ('lionmuon_257m_' + schedule + '_full')
            if directory.is_symlink():
                raise RuntimeError('Refusing symlink source')
            files = [directory / 'metrics.jsonl'] + sorted(directory.glob('*.json'))
            for path in files:
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError('Missing regular source file: ' + str(path))
                if path.stat().st_size > 20_000_000:
                    raise RuntimeError('Unexpected source size')
                raw = path.read_bytes()
                encoded = base64.b64encode(zlib.compress(raw)).decode()
                parts = [encoded[i:i + 16000] for i in range(0, len(encoded), 16000)]
                for index, part in enumerate(parts):
                    print('SOURCE_CHUNK=' + json.dumps(dict(schedule=schedule, path=str(path),
                          sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                          index=index, parts=len(parts), data=part)), flush=True)
        guard()
        print('LIONMUON_EXPORT_COMPLETE', flush=True)
    finally:
        stop.set()


if __name__ == '__main__':
    main()
