"""Locate existing plain-Muon logs and export validation records, read-only."""

import json
import os
from pathlib import Path
import re


def main() -> None:
    for root in (Path('/home/jovyan/dimativator'), Path('/workspace-SR006.nfs2/dimativator'),
                 Path('/workspace-SR006.nfs3/dimativator')):
        if not root.exists():
            continue
        for directory, folders, files in os.walk(root):
            folders[:] = [name for name in folders if name not in ('.git', '.cache', 'datasets', 'ckpts', 'checkpoints')]
            for filename in files:
                if not (filename.endswith('.jsonl') or filename.endswith('.log')):
                    continue
                path = Path(directory) / filename
                if not re.search(r'(^|[/_\-])muon([/_\-.]|$)', str(path).lower()):
                    continue
                if path.stat().st_size > 100 * 10**6:
                    continue
                rows = {}
                for line in path.read_text(errors='replace').splitlines():
                    if 'METRIC_JSON=' in line:
                        line = line.split('METRIC_JSON=', 1)[1]
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if not isinstance(row, dict) or row.get('event') != 'validation':
                        continue
                    if 'val/loss' not in row and 'final-val/loss' not in row:
                        continue
                    rows[row.get('iter')] = {k: v for k, v in row.items()
                        if k in ('iter', 'consumed_tokens', 'val/loss', 'final-val/loss', 'experiment', 'group')}
                if rows:
                    print('MUON_BASELINE=' + json.dumps(dict(path=str(path), validation=list(rows.values()),
                          sibling_files=sorted(p.name for p in path.parent.iterdir())[:20])), flush=True)


if __name__ == '__main__':
    main()
