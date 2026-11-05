"""Gateway controller: four 257M trials, then one best-LR 500M trial.

Run with system Python on brain_lab. W&B credentials stay in the gateway's
netrc. Submission receipts are exclusive and never retried automatically
after an ambiguous scheduler response. No training runs on the gateway.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import netrc
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

BRANCH = 'codex/soft-lion-muon-cloud-20261105'
REPO = 'https://github.com/dimativator/H-MoE-Part-cloud.git'
ROOT = '/workspace-SR006.nfs3/dimativator/soft-lion-muon-20261105'
LOGS = '/home/jovyan/shares/SR006.nfs3/mlsub-logs'
GATEWAY = Path('/home/dimativator/soft-lion-muon-20261105')
LRS = ('2e-3', '1e-3', '5e-4', '1e-4')
CONFIG_KEYS = ('opt', 'lr', 'weight_decay', 'iterations', 'seed', 'data_seed', 'dtype',
    'model_parameter_dtype', 'batch_size', 'acc_steps', 'sequence_length', 'no_local_save',
    'latest_ckpt_interval', 'permanent_ckpt_interval', 'inter_ckpts', 'n_layer', 'n_embd',
    'soft_lion_muon_period', 'soft_lion_muon_beta1', 'soft_lion_muon_beta2',
    'soft_lion_muon_lr_ratio', 'soft_lion_muon_start_fraction', 'soft_lion_muon_eps',
    'soft_lion_muon_ns_steps', 'soft_lion_muon_soft_ns_steps', 'soft_lion_muon_aux_lr')


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def command(args: list[str], timeout: int = 40) -> dict:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired:
        return dict(error='timeout')


def run_name(size: int, lr: str) -> str:
    wd, world = ('1e-1', 2) if size == 257 else ('1e-4', 4)
    return f'adapt-soft_lion_muon_{size}m_bf16_lr{lr}_wd{wd}_1xC_{world}gpu-20261105'


def wandb_run(size: int, lr: str) -> dict:
    auth = netrc.netrc().authenticators('wandb-radfan.ru')
    if not auth:
        raise RuntimeError('Online W&B credentials unavailable')
    query = ('query { project(name:"fp8-pretrain", entityName:"andrey") { run(name:"'
             + run_name(size, lr) + '") { name state summaryMetrics config } } }')
    token = base64.b64encode(('api:' + auth[2]).encode()).decode()
    request = urllib.request.Request('https://wandb-radfan.ru/graphql',
        data=json.dumps(dict(query=query)).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Basic ' + token})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)
    if data.get('errors'):
        raise RuntimeError('W&B query failed')
    record = (data.get('data', {}).get('project') or {}).get('run')
    if not record:
        return {}
    record['summary'] = json.loads(record.pop('summaryMetrics'))
    config = json.loads(record.pop('config'))
    record['config'] = {key: config.get(key, {}).get('value') for key in CONFIG_KEYS}
    record['url'] = 'https://wandb-radfan.ru/andrey/fp8-pretrain/runs/' + run_name(size, lr)
    return record


def collect(label: str, size: int, lr: str, logs: bool = False) -> dict:
    receipt = json.loads((GATEWAY / (label + '.json')).read_text())
    record = dict(size=size, lr=lr, job_id=receipt.get('job_id'), timestamp=time.time())
    if not record['job_id']:
        record['error'] = 'Submission receipt has no job ID'
        return record
    raw = command(['mlsub', 'status', record['job_id']])
    try:
        record['status'] = json.loads(raw['stdout'])
    except (KeyError, ValueError):
        record['status_error'] = raw.get('error', 'Invalid scheduler status')
    try:
        record['wandb'] = wandb_run(size, lr)
    except Exception as exc:
        record['wandb_error'] = type(exc).__name__
    if logs:
        raw = command(['mlsub', 'logs', record['job_id'], '--tail', '100'], timeout=25)
        events = {}
        for line in raw.get('stdout', '').splitlines():
            for marker in ('LIVE_STATUS=', 'LAUNCHER_DISK_GUARD=', 'RUN_VERIFIED='):
                if marker in line:
                    try:
                        value = json.loads(line.split(marker, 1)[1])
                        if marker != 'LIVE_STATUS=' or value.get('phase') == '1xC':
                            events[marker[:-1]] = value
                    except ValueError:
                        pass
        record['events'] = events
    return record


def validate_final(record: dict, size: int, lr: str) -> float:
    status = record.get('status', {})
    run = record.get('wandb', {})
    target, layers, width = (39250, 12, 1024) if size == 257 else (75457, 18, 1280)
    if status.get('status', '').lower() != 'completed' or status.get('error_code') != 0:
        raise ValueError('Scheduler has not confirmed successful completion')
    if run.get('state') != 'finished':
        raise ValueError('W&B has not confirmed a finished run')
    expected = dict(opt='soft_lion_muon', lr=float(lr), weight_decay=0.1 if size == 257 else 0.0001,
        iterations=target, seed=0, data_seed=1337, dtype='bfloat16', model_parameter_dtype='float32',
        batch_size=16 if size == 257 else 8, acc_steps=4, sequence_length=1024,
        no_local_save=True, latest_ckpt_interval=0, permanent_ckpt_interval=0, inter_ckpts=[],
        n_layer=layers, n_embd=width, soft_lion_muon_period=5, soft_lion_muon_beta1=0.95,
        soft_lion_muon_beta2=0.99, soft_lion_muon_lr_ratio=0.01,
        soft_lion_muon_start_fraction=0.9, soft_lion_muon_eps=1e-4,
        soft_lion_muon_ns_steps=6, soft_lion_muon_soft_ns_steps=30, soft_lion_muon_aux_lr=0.001)
    mismatches = {k: (run.get('config', {}).get(k), v) for k, v in expected.items()
                  if run.get('config', {}).get(k) != v}
    if mismatches:
        raise RuntimeError('Final configuration mismatch: ' + str(mismatches))
    summary = run.get('summary', {})
    if summary.get('iter') != target or summary.get('consumed_tokens') != target * 128 * 1024:
        raise RuntimeError('Final token/step budget mismatch')
    loss = summary.get('final-val/loss')
    if not isinstance(loss, (float, int)) or not math.isfinite(loss):
        raise RuntimeError('Missing or nonfinite final validation loss')
    return float(loss)


def select_winner(records: list[dict]) -> dict:
    by_lr = {r['lr']: r for r in records}
    if len(records) != 4 or set(by_lr) != set(LRS):
        raise RuntimeError('All four authorized trials are required')
    losses = {lr: validate_final(by_lr[lr], 257, lr) for lr in LRS}
    best = min(LRS, key=lambda lr: (losses[lr], float(lr)))
    return dict(lr=best, final_val_loss=losses[best], all_final_losses=losses,
                criterion='lowest_final_validation_loss_after_full_1xC', timestamp=time.time())


def check_policy() -> None:
    if json.loads(Path('/etc/mlsub/policy.json').read_text())['logs_dir'] != LOGS:
        raise RuntimeError('Bootstrap logs no longer use the verified NFS3 mount')


def submit(label: str, revision: str, *, size: int = 257, lr: str | None = None) -> dict:
    check_policy()
    path = GATEWAY / (label + '.json')
    if path.exists():
        previous = json.loads(path.read_text())
        if not previous.get('job_id'):
            raise RuntimeError('Ambiguous previous submission requires reconciliation: ' + label)
        return previous
    preflight = lr is None
    proof = None
    if not preflight:
        proof = json.loads((GATEWAY / f'preflight{size}_verified.json').read_text())
        if not 0 <= time.time() - proof['timestamp'] <= 600 or proof['code_revision'] != revision:
            raise RuntimeError('Storage preflight is stale or uses another code revision')
        if any(free <= 5_100_000_000 for free in proof['available_bytes'].values()):
            raise RuntimeError('Storage reserve unavailable before launch')
        if lr not in LRS:
            raise RuntimeError('Unauthorized LR')
    inventory = command(['mlsub', 'list'])
    if inventory.get('returncode') != 0:
        raise RuntimeError('Scheduler inventory unavailable')
    note = 'soft_lion_muon_' + label + '_20261105'
    if note in inventory['stdout']:
        raise RuntimeError('Matching scheduler job exists without a receipt')
    auth = netrc.netrc().authenticators('wandb-radfan.ru')
    if not auth:
        raise RuntimeError('Online W&B is mandatory')
    key = auth[2]
    entry = 'scripts/cloud/inspect_soft_lion_muon.py' if preflight else 'scripts/cloud/run_soft_lion_muon.py'
    world = 0 if preflight else (2 if size == 257 else 4)
    env = dict(CODE_REVISION=revision, RESULTS_DIR=ROOT if preflight else ROOT + f'/{size}m/lr{lr}',
               WANDB_MODE='online', WANDB_API_KEY=key, VERIFIED_BOOTSTRAP_LOGS_DIR=LOGS,
               CLOUD_DISABLE_CHECKPOINTS='1')
    if preflight:
        env.update(SOFT_LION_MUON_PREFLIGHT='1', CHECK_PACKED_DATA='1', CHECK_WANDB='1')
    else:
        env.update(LR=lr, MODEL_SIZE=str(size), CLOUD_TRAIN_WORLD_SIZE=str(world))
    args = ['mlsub', 'run', '--repo', REPO, '--branch', BRANCH, '--entry', entry,
            '--gpus', 'cpu' if preflight else str(world), '--image', 'efficient', '--no-pip', '--note', note]
    for name, value in env.items():
        args += ['--env', name + '=' + value]
    record = dict(state='submitting', timestamp=time.time(), label=label, size=size, lr=lr,
                  code_revision=revision, gpus=world, results_dir=env['RESULTS_DIR'], proof=proof)
    with path.open('x') as stream:
        json.dump(record, stream)
    try:
        # The gateway's HTTP connect/read timeout is 600s. Let it report its
        # actual result rather than orphaning a still-pending submission.
        result = subprocess.run(args, capture_output=True, text=True, timeout=660)
        record.update(returncode=result.returncode, stdout=result.stdout.replace(key, '[REDACTED]'),
                      stderr=result.stderr.replace(key, '[REDACTED]'))
        match = re.search(r'lm-mpi-job-[0-9a-f-]+', record['stdout'])
        if match:
            record.update(state='submitted', job_id=match.group())
        else:
            record['state'] = 'ambiguous_submission'
    except subprocess.TimeoutExpired:
        record['state'] = 'ambiguous_submission'
    atomic_json(path, record)
    if not record.get('job_id'):
        raise RuntimeError('Submission not confirmed: ' + label)
    return record


def preflight(size: int, revision: str) -> dict | None:
    receipt = submit(f'preflight{size}', revision, size=size)
    raw = command(['mlsub', 'status', receipt['job_id']])
    if raw.get('returncode') != 0:
        return None
    status = json.loads(raw['stdout'])
    state = status.get('status', '').lower()
    if state in ('failed', 'cancelled', 'canceled', 'killed'):
        raise RuntimeError('Storage/regression preflight failed: ' + str(status))
    if state != 'completed':
        return None
    logs = command(['mlsub', 'logs', receipt['job_id'], '--tail', '300'])
    text = logs.get('stdout', '')
    if 'SOFT_LION_MUON_REGRESSION_OK' not in text:
        return None
    events = {}
    for line in text.splitlines():
        for marker in ('SPACE_SNAPSHOT=', 'PACKED_DATA=', 'WANDB_READY='):
            if marker in line:
                events[marker[:-1]] = json.loads(line.split(marker, 1)[1])
    if not all(k in events for k in ('SPACE_SNAPSHOT', 'PACKED_DATA', 'WANDB_READY')):
        return None
    proof = events['SPACE_SNAPSHOT']
    if proof['logs_dir'] != LOGS or not events['WANDB_READY']['ready']:
        raise RuntimeError('Bootstrap log path or W&B preflight failed')
    if any(free <= 5_100_000_000 for free in proof['available_bytes'].values()):
        raise RuntimeError('КРИТИЧЕСКИ МАЛО МЕСТА НА CLOUD.RU: НОВЫЕ ЗАПУСКИ ОСТАНОВЛЕНЫ')
    metadata = events['PACKED_DATA']
    if (metadata['format'] != 'packed_fineweb_h200_v3' or metadata['world_size'] != 2
            or metadata['batch_size'] != 16 or metadata['sequence_length'] != 1024
            or metadata['iterations'] < 75457):
        raise RuntimeError('Packed data configuration mismatch')
    proof.update(code_revision=revision, regression_passed=True, packed_data=metadata,
                 nfs2_write_usage=False)
    atomic_json(GATEWAY / f'preflight{size}_verified.json', proof)
    return proof


def snapshot(logs: bool = False) -> dict:
    trials = [(f'257m_lr{lr}', 257, lr) for lr in LRS if (GATEWAY / f'257m_lr{lr}.json').exists()]
    if (GATEWAY / '500m.json').exists():
        receipt = json.loads((GATEWAY / '500m.json').read_text())
        trials.append(('500m', 500, receipt['lr']))
    with ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(lambda args: collect(*args, logs=logs), trials))
    payload = dict(timestamp=time.time(), trials=records)
    atomic_json(GATEWAY / 'latest_snapshot.json', payload)
    return payload


def advance(revision: str) -> dict:
    state_path = GATEWAY / 'controller_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(stage='preflight257')
    stage = state['stage']
    if stage == 'preflight257':
        if preflight(257, revision):
            state['stage'] = 'submit257'
    elif stage == 'submit257':
        # Independent, exclusively recorded submissions share the same fresh
        # storage proof. Slow API responses must not age it for later trials.
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda lr: submit(f'257m_lr{lr}', revision, lr=lr), LRS))
        state['stage'] = 'sweep257'
    elif stage in ('sweep257', 'run500'):
        records = snapshot()['trials']
        for record in records:
            status = record.get('status', {})
            scheduler = status.get('status', '').lower()
            if scheduler in ('failed', 'cancelled', 'canceled', 'killed') or (
                    scheduler == 'completed' and status.get('error_code') != 0):
                raise RuntimeError('Tracked training job did not complete successfully: ' + record['job_id'])
        if stage == 'sweep257':
            sweep = [r for r in records if r['size'] == 257]
            if len(sweep) == 4 and all(r.get('status', {}).get('status', '').lower() == 'completed'
                    and r.get('wandb', {}).get('state') == 'finished' for r in sweep):
                winner = select_winner(sweep)
                atomic_json(GATEWAY / 'winner.json', winner)
                state.update(stage='preflight500', winner=winner)
        else:
            final = next(r for r in records if r['size'] == 500)
            if (final.get('status', {}).get('status', '').lower() == 'completed'
                    and final.get('wandb', {}).get('state') == 'finished'):
                validate_final(final, 500, state['winner']['lr'])
                atomic_json(GATEWAY / '500m_verified_final.json', final)
                state['stage'] = 'complete'
    elif stage == 'preflight500':
        if preflight(500, revision):
            state['stage'] = 'submit500'
    elif stage == 'submit500':
        # Recheck all four final records immediately before selecting/submitting.
        winner = select_winner([r for r in snapshot()['trials'] if r['size'] == 257])
        if winner['lr'] != state['winner']['lr']:
            raise RuntimeError('Winner changed after selection')
        submit('500m', revision, size=500, lr=winner['lr'])
        state['stage'] = 'run500'
    state.update(timestamp=time.time(), code_revision=revision)
    atomic_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('watch', 'once', 'snapshot'))
    parser.add_argument('--revision', default='')
    parser.add_argument('--logs', action='store_true')
    args = parser.parse_args()
    GATEWAY.mkdir(exist_ok=True)
    if args.mode == 'snapshot':
        print(json.dumps(snapshot(logs=args.logs)))
        return
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        raise ValueError('A pinned full code revision is required')
    with (GATEWAY / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                state = advance(args.revision)
                print('CONTROLLER_STATE=' + json.dumps(state), flush=True)
                if state['stage'] == 'complete' or args.mode == 'once':
                    return
            except RuntimeError as exc:
                atomic_json(GATEWAY / 'controller_failure.json', dict(timestamp=time.time(), error=str(exc)))
                raise
            except Exception as exc:
                # API/network failures do not authorize retries of training submissions.
                print('CONTROLLER_TRANSIENT=' + type(exc).__name__, flush=True)
                if args.mode == 'once':
                    raise
            time.sleep(30)


if __name__ == '__main__':
    main()
