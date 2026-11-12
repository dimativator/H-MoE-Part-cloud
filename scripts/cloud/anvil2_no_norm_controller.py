"""Gateway receipts and verified stage transition for ANVIL II without normalization."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import netrc
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from anvil2_controller import LOGS, REPO, atomic_json, check_policy, command, CONFIG_KEYS

BRANCH = 'codex/anvil2-no-norm-cloud-20261112'
ROOT = '/workspace-SR006.nfs3/dimativator/anvil2-no-norm-20261112'
GATEWAY = Path('/home/dimativator/anvil2-no-norm-20261112')
LRS = ('1e-2', '5e-3', '2e-3', '1e-3', '5e-4', '1e-4')


def run_id(size: int, lr: str) -> str:
    wd, world = ('1e-1', 2) if size == 257 else ('1e-4', 4)
    return f'adapt-anvil2_no_norm_{size}m_bf16_lr{lr}_wd{wd}_1xC_{world}gpu-20261112'


def submit(revision: str, label: str, *, size: int = 257, lr: str | None = None,
           preflight: bool = False) -> dict:
    check_policy()
    if size not in (257, 500) or (lr is not None and lr not in LRS):
        raise ValueError('Unauthorized stage or LR')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', label):
        raise ValueError('Invalid receipt label')
    path = GATEWAY / (label + '.json')
    if path.exists():
        prior = json.loads(path.read_text())
        if (not prior.get('job_id') or prior.get('code_revision') != revision
                or prior.get('size') != size or prior.get('lr') != lr
                or prior.get('preflight') != preflight or prior.get('label') != label):
            raise RuntimeError('Existing or ambiguous receipt requires reconciliation: ' + label)
        return prior
    if lr is not None:
        proof = json.loads((GATEWAY / f'preflight{size}_verified.json').read_text())
        if proof['code_revision'] != revision or not 0 <= time.time() - proof['timestamp'] <= 600:
            raise RuntimeError('Fresh storage/source proof required')
        if any(free <= 5_500_000_000 for free in proof['available_bytes'].values()):
            raise RuntimeError('Insufficient parallel log/cache storage reserve')
        if size == 500:
            selected = json.loads((GATEWAY / 'selection.json').read_text())
            if selected['code_revision'] != revision or selected['lr'] != lr:
                raise RuntimeError('500M LR must be the verified sweep winner')
    note = 'anvil2_no_norm_' + label + '_20261112'
    inventory = command(['mlsub', 'list', '--active'])
    if inventory['returncode'] or note in inventory['stdout']:
        raise RuntimeError('Cannot rule out an existing job: ' + label)
    env = dict(CODE_REVISION=revision, RESULTS_DIR=ROOT, ANVIL2_NO_NORMALIZATION='1',
               VERIFIED_BOOTSTRAP_LOGS_DIR=LOGS, CLOUD_DISABLE_CHECKPOINTS='1')
    key = ''
    if preflight or lr is not None:
        auth = netrc.netrc().authenticators('wandb-radfan.ru')
        if not auth:
            raise RuntimeError('Online W&B credentials unavailable')
        key = auth[2]
        env.update(WANDB_MODE='online', WANDB_API_KEY=key)
    if preflight:
        env.update(ANVIL2_PREFLIGHT='1', CHECK_PACKED_DATA='1', CHECK_WANDB='1')
    if lr is not None:
        env.update(RESULTS_DIR=ROOT + f'/{size}m/lr{lr}', LR=lr, ANVIL2_STAGE=f'{size}m',
                   CLOUD_TRAIN_WORLD_SIZE='2' if size == 257 else '4')
    args = ['mlsub', 'run', '--repo', REPO, '--branch', BRANCH, '--entry',
            'scripts/cloud/run_anvil2_no_norm.py' if lr is not None else 'scripts/cloud/inspect_anvil2.py',
            '--gpus', ('2' if size == 257 else '4') if lr is not None else 'cpu',
            '--image', 'efficient', '--no-pip', '--note', note]
    for name, value in env.items():
        args += ['--env', name + '=' + value]
    try:
        dry = command(args + ['--dry-run'])
    except subprocess.TimeoutExpired:
        raise RuntimeError('Dry-run timed out before submission') from None
    if dry['returncode']:
        raise RuntimeError('Dry-run failed')
    record = dict(state='submitting', timestamp=time.time(), label=label, size=size, lr=lr,
                  code_revision=revision, results_dir=env['RESULTS_DIR'], preflight=preflight,
                  gpus=(2 if size == 257 else 4) if lr is not None else 0)
    with path.open('x') as stream:
        json.dump(record, stream)
    try:
        result = command(args, timeout=660)
        record.update({k: v.replace(key, '[REDACTED]') if key and isinstance(v, str) else v
                       for k, v in result.items()})
        match = re.search(r'lm-mpi-job-[0-9a-f-]+', record['stdout'])
        record['state'] = 'submitted' if match else 'ambiguous_submission'
        if match:
            record['job_id'] = match.group()
    except subprocess.TimeoutExpired:
        record['state'] = 'ambiguous_submission'
    atomic_json(path, record)
    if not record.get('job_id'):
        raise RuntimeError('Submission is ambiguous; do not retry: ' + label)
    return record


def inspection(label: str) -> dict:
    receipt = json.loads((GATEWAY / (label + '.json')).read_text())
    status = json.loads(command(['mlsub', 'status', receipt['job_id']])['stdout'])
    if status.get('status', '').lower() != 'completed':
        return dict(ready=False, status=status)
    if status.get('error_code') != 0:
        raise RuntimeError('Inspector failed: ' + receipt['job_id'])
    result = command(['mlsub', 'logs', receipt['job_id'], '--tail', '500'])
    if result['returncode']:
        raise RuntimeError('Inspector logs unavailable')
    log = re.sub(r'\[\d+,\d+\]<(?:stdout|stderr)>:', '', result['stdout'])
    (GATEWAY / (label + '.log')).write_text(log)
    events = {}
    for line in log.splitlines():
        for marker in ('SPACE_SNAPSHOT', 'PACKED_DATA', 'WANDB_READY', 'NFS2_READ_ONLY',
                       'PROGRESS', 'FINAL', 'WORKER_DISK'):
            if marker + '=' in line:
                events.setdefault(marker, []).append(json.loads(line.split(marker + '=', 1)[1]))
    if not events.get('SPACE_SNAPSHOT'):
        raise RuntimeError('No current filesystem telemetry')
    payload = dict(ready=True, job_id=receipt['job_id'], code_revision=receipt['code_revision'],
                   status=status, events=events)
    atomic_json(GATEWAY / (label + '_observed.json'), payload)
    if not receipt['preflight']:
        atomic_json(GATEWAY / 'latest_inspection.json', payload)
    return {**payload, 'regression_passed': 'ANVIL2_REGRESSION_OK' in log}


def verify_preflight(size: int, revision: str) -> dict:
    observed = inspection(f'preflight{size}')
    if not observed['ready']:
        return observed
    if observed['code_revision'] != revision:
        raise RuntimeError('Preflight receipt uses another source revision')
    events = observed['events']
    if not observed['regression_passed'] or not events.get('WANDB_READY', [{}])[-1].get('ready'):
        raise RuntimeError('Preflight regression or W&B connectivity failure')
    proof = events['SPACE_SNAPSHOT'][-1]
    if proof['logs_dir'] != LOGS or any(x <= 5_500_000_000 for x in proof['available_bytes'].values()):
        raise RuntimeError('Preflight disk reserve unavailable')
    data = events['PACKED_DATA'][-1]
    expected = dict(format='packed_fineweb_h200_v3', world_size=2, batch_size=16, sequence_length=1024)
    if any(data[k] != v for k, v in expected.items()) or data['iterations'] < 75457:
        raise RuntimeError('Packed data mismatch')
    proof.update(ready=True, code_revision=revision, regression_passed=True,
                 packed_data=data, nfs2_write_usage=False)
    atomic_json(GATEWAY / f'preflight{size}_verified.json', proof)
    return proof


def snapshot() -> dict:
    receipts = [json.loads(p.read_text()) for p in sorted(GATEWAY.glob('257m_lr*.json'))]
    if (GATEWAY / '500m.json').exists():
        receipts.append(json.loads((GATEWAY / '500m.json').read_text()))
    def status(receipt: dict) -> dict:
        value = dict(size=receipt['size'], lr=receipt['lr'], job_id=receipt.get('job_id'))
        if not value['job_id']:
            return value
        try:
            value['status'] = json.loads(command(['mlsub', 'status', value['job_id']], timeout=25)['stdout'])
        except (subprocess.TimeoutExpired, ValueError):
            value['status_error'] = 'Scheduler read unavailable'
        return value
    with ThreadPoolExecutor(max_workers=7) as pool:
        records = list(pool.map(status, receipts))
    if records:
        auth = netrc.netrc().authenticators('wandb-radfan.ru')
        if not auth:
            raise RuntimeError('W&B credentials unavailable')
        query = 'query { project(name:"fp8-pretrain",entityName:"andrey") {'
        for i, record in enumerate(records):
            query += f' r{i}:run(name:"{run_id(record["size"], record["lr"])}") {{ state config summaryMetrics }}'
        query += ' }}'
        token = base64.b64encode(('api:' + auth[2]).encode()).decode()
        request = urllib.request.Request('https://wandb-radfan.ru/graphql',
            data=json.dumps(dict(query=query)).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Basic ' + token})
        with urllib.request.urlopen(request, timeout=25) as response:
            data = json.load(response)
        if data.get('errors'):
            raise RuntimeError('W&B batch query failed')
        runs = (data.get('data') or {}).get('project') or {}
        for i, record in enumerate(records):
            run = runs.get('r' + str(i))
            if run:
                config = json.loads(run['config'])
                keys = (*CONFIG_KEYS, 'anvil2_no_normalization')
                record['wandb'] = dict(state=run['state'], summary=json.loads(run['summaryMetrics']),
                    config={k: config.get(k, {}).get('value') for k in keys})
                record['wandb_url'] = 'https://wandb-radfan.ru/andrey/fp8-pretrain/runs/' + run_id(record['size'], record['lr'])
    payload = dict(timestamp=time.time(), trials=records)
    atomic_json(GATEWAY / 'latest_snapshot.json', payload)
    return payload


def validate_final(record: dict, raw: dict) -> float:
    size, lr = record['size'], record['lr']
    target, layers, width, heads, world, batch, wd = ((39250, 12, 1024, 8, 2, 16, 0.1)
        if size == 257 else (75457, 18, 1280, 20, 4, 8, 0.0001))
    state, run = record.get('status', {}), record.get('wandb', {})
    if state.get('status', '').lower() != 'completed' or state.get('error_code') != 0 or run.get('state') != 'finished':
        raise RuntimeError('Scheduler and W&B must confirm successful completion')
    expected = dict(opt='anvil2', lr=float(lr), weight_decay=wd, iterations=target, n_layer=layers,
        n_embd=width, n_head=heads, batch_size=batch, acc_steps=4, sequence_length=1024,
        seed=0, data_seed=1337, dtype='bfloat16', model_parameter_dtype='float32', beta1=0.9,
        beta2=0.95, scheduler='wsd', warmup_steps=2000, wsd_fract_decay=0.1,
        wsd_final_lr_scale=0, decay_type='cosine', no_local_save=True, latest_ckpt_interval=0,
        permanent_ckpt_interval=0, inter_ckpts=[], anvil2_no_normalization=True,
        anvil2_engage_fraction=0.5, anvil2_momentum_warmup_steps=240, anvil2_momentum_cooldown_steps=50)
    if any(run['config'].get(k) != v for k, v in expected.items()):
        raise RuntimeError('Final W&B configuration differs from the authorized experiment')
    summary = run['summary']
    loss = summary.get('final-val/loss')
    if not isinstance(loss, (float, int)) or not math.isfinite(loss):
        raise RuntimeError('Finite final validation loss required')
    if summary.get('iter') != target or summary.get('consumed_tokens') != target * 128 * 1024:
        raise RuntimeError('Incomplete step or token budget')
    matching = [f for f in raw['events'].get('FINAL', []) if f.get('phase') == '1xC'
                and f.get('size') == size and f.get('lr') == lr]
    expected_raw = dict(iter=target, consumed_tokens=target * 128 * 1024, algorithm='ANVIL_II_NO_NORM',
        matrix_normalization=False, weight_decay=wd, auxiliary_lr=float(lr), auxiliary_weight_decay=wd,
        physical_world_size=world, checkpoint_saving=False, engage_step=math.ceil(target / 2),
        ns_steps=6, **{'final-val/loss': loss})
    if len(matching) != 1 or any(matching[0].get(k) != v for k, v in expected_raw.items()):
        raise RuntimeError('Matching completed worker artifact required')
    return loss


def select_winner(revision: str) -> dict:
    records = [r for r in snapshot()['trials'] if r['size'] == 257]
    raw = json.loads((GATEWAY / 'latest_inspection.json').read_text())
    if raw['code_revision'] != revision or len(records) != 6 or {r['lr'] for r in records} != set(LRS):
        raise RuntimeError('All six authorized trials and pinned raw artifacts are required')
    losses = {r['lr']: validate_final(r, raw) for r in records}
    lr = min(LRS, key=lambda x: (losses[x], float(x)))
    selection = dict(timestamp=time.time(), code_revision=revision, lr=lr,
                     final_val_loss=losses[lr], final_losses=losses,
                     criterion='lowest_final_validation_loss_after_verified_1xC')
    atomic_json(GATEWAY / 'selection.json', selection)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'verify', 'launch', 'snapshot',
                                          'inspect', 'inspection', 'select', 'followup'))
    parser.add_argument('--revision', required=True)
    parser.add_argument('--size', type=int, choices=(257, 500), default=257)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-f0-9]{40}', args.revision):
        raise ValueError('A pinned full source revision is required')
    GATEWAY.mkdir(parents=True, exist_ok=True)
    with (GATEWAY / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'preflight':
            value = submit(args.revision, f'preflight{args.size}', size=args.size, preflight=True)
        elif args.action == 'verify':
            value = verify_preflight(args.size, args.revision)
        elif args.action == 'launch':
            with ThreadPoolExecutor(max_workers=6) as pool:
                value = list(pool.map(lambda lr: submit(args.revision, '257m_lr' + lr, lr=lr), LRS))
            atomic_json(GATEWAY / 'submissions257.json', dict(timestamp=time.time(), trials=value))
        elif args.action == 'snapshot':
            value = snapshot()
        elif args.action == 'inspect':
            if not args.label.startswith('inspection'):
                raise ValueError('Use a unique inspection-prefixed label')
            value = submit(args.revision, args.label)
        elif args.action == 'inspection':
            value = inspection(args.label)
        elif args.action == 'select':
            value = select_winner(args.revision)
        else:
            selected = select_winner(args.revision)
            value = submit(args.revision, '500m', size=500, lr=selected['lr'])
        print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
