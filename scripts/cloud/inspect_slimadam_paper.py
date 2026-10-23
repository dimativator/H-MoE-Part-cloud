"""Read saved SlimAdam sweep metrics without modifying training artifacts."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil

from run_slimadam_paper import LRS, TOKENS, build_command, verify_final


def read_metrics(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return records


def log_tail(path: Path) -> dict:
    if not path.exists():
        return dict(exists=False)
    stat = path.stat()
    with path.open('rb') as stream:
        stream.seek(max(0, stat.st_size - 32768))
        tail = stream.read().decode(errors='replace').splitlines()[-20:]
    return dict(exists=True, bytes=stat.st_size, modified_utc=datetime.fromtimestamp(
                stat.st_mtime, timezone.utc).isoformat(), tail='\n'.join(tail)[-5000:])


def main() -> None:
    date = os.environ.get('CAMPAIGN_DATE', '20261023')
    root = Path(os.environ.get('RESULTS_DIR', f'/workspace-SR006.nfs2/dimativator/slimadam-paper-{date}'))
    snapshot = dict(checked_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
                    root_exists=root.exists(), runs=[], free_bytes={})
    for mount in ('/home/jovyan', '/workspace-SR006.nfs2', '/workspace-SR006.nfs3'):
        if Path(mount).exists():
            snapshot['free_bytes'][mount] = shutil.disk_usage(mount).free
    for lr in LRS:
        _, path, name = build_command(lr, root, date)
        records = read_metrics(path)
        train = [r for r in records if r.get('event') == 'train']
        validation = [r for r in records if r.get('event') == 'validation']
        record = dict(lr=lr, name=name, metrics_path=str(path), metrics_exist=path.exists(),
                      recent_train=train[-10:], recent_validation=validation[-3:],
                      metric_count=len(records), final=None, config=None, logs={})
        config = root / 'claims' / f'{name}_config.json'
        if config.exists():
            saved = json.loads(config.read_text())
            record['config'] = {key: saved.get(key) for key in (
                'code_revision', 'lr', 'weight_decay', 'betas', 'parameters',
                'second_moment_elements', 'hidden_matrices_compressed', 'layer_map_sha256',
                'iterations', 'tokens', 'global_batch', 'checkpoint_saving')}
            record['config_matches'] = (
                saved.get('layer_map_sha256') == 'bfac85e4bba50eb31e752f55418622dfec6dae41b8cf1f9832f8c40dd18bc968'
                and saved.get('hidden_matrices_compressed') == 84
                and saved.get('second_moment_elements') == 233728
                and saved.get('lr') == float(lr) and saved.get('weight_decay') == 0.1
                and saved.get('betas') == [0.9, 0.99] and saved.get('tokens') == TOKENS
                and saved.get('checkpoint_saving') is False)
        final = root / 'finals' / f'{name}.json'
        if final.exists():
            record['final'] = json.loads(final.read_text())
            try:
                record['verified_native_final'] = verify_final(path)
            except (RuntimeError, FileNotFoundError) as error:
                record['final_verification_error'] = str(error)
        record['nonfinite_metrics'] = [r for r in records if any(
            isinstance(value, (int, float)) and not math.isfinite(value)
            for value in r.values())][-3:]
        for rank in (0, 1):
            record['logs'][str(rank)] = log_tail(root / 'logs' / f'{name}_rank{rank}.log')
        snapshot['runs'].append(record)
    print('SLIMADAM_PAPER_SNAPSHOT=' + json.dumps(snapshot, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
