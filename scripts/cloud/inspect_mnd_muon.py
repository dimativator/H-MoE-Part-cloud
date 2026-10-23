"""Read the four LR runs without modifying their files or scheduler state."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from run_mnd_muon import LRS, build_command


def main() -> None:
    date = os.environ.get('CAMPAIGN_DATE', '20261022')
    root = Path(os.environ.get('RESULTS_DIR', f'/workspace-SR006.nfs2/dimativator/mnd-muon-{date}'))
    snapshot = dict(checked_utc=datetime.now(timezone.utc).isoformat(), runs=[], disk_free_bytes={})
    for mount in ('/home/jovyan', '/workspace-SR006.nfs2', '/workspace-SR006.nfs3'):
        if Path(mount).exists():
            snapshot['disk_free_bytes'][mount] = shutil.disk_usage(mount).free
    for lr in LRS:
        _, path, name = build_command(lr, root, date)
        record = dict(lr=float(lr), name=name, metrics_path=str(path), validation=[], latest_train=None)
        if path.exists():
            lines = path.read_text().splitlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    metric = json.loads(line)
                except json.JSONDecodeError:
                    if index != len(lines) - 1:
                        raise
                    continue  # Writer may be appending the final line.
                if metric.get('event') == 'validation':
                    record['validation'].append(metric)
                elif metric.get('event') == 'train':
                    record['latest_train'] = metric
        final = root / 'finals' / f'{name}.json'
        record['verified_final'] = json.loads(final.read_text()) if final.exists() else None
        snapshot['runs'].append(record)
    print('MND_MUON_SNAPSHOT=' + json.dumps(snapshot, sort_keys=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
