"""CPU-only environment, dataset and storage checks before the GPU sweep."""

import json
from pathlib import Path
import os
import shutil
import subprocess
import sys


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[2])
    root = Path('/workspace-SR006.nfs2/dimativator')
    data = Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed')
    metadata = json.loads((data / 'packed_metadata.json').read_text())
    assert metadata['format'] == 'packed_fineweb_h200_v3'
    assert metadata['world_size'] == 2 and metadata['batch_size'] == 16
    assert metadata['iterations'] >= 39250
    for item in metadata['ranks']:
        for segment in item.get('segments', [item]):
            path = data / segment['file']
            assert path.is_file() and path.stat().st_size == segment['bytes'], str(path)
    assert shutil.disk_usage(root).free >= 10**9
    env = {**os.environ, 'PYTHONPATH': 'src:.'}
    for pattern in ('test_mnd_muon.py', 'test_adaptive_muon*.py', 'test_fineweb*replay*.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v'],
                       env=env, check=True)
    print('PREFLIGHT_VERIFIED=' + json.dumps(dict(data=str(data), format=metadata['format'],
          free_bytes={str(p): shutil.disk_usage(p).free for p in (root, data)},
          tests=24, checkpoint_saving=False)), flush=True)


if __name__ == '__main__':
    main()
