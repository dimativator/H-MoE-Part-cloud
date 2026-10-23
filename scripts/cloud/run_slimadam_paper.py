"""One SlimAdam paper-map LR per two-GPU Cloud.ru job."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


LRS = ('2e-3', '1e-3', '5e-4', '1e-4')
ITERATIONS = 39250
TOKENS = ITERATIONS * 128 * 1024
DATA = Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed')
LAYER_MAP = Path('src/optim/memory_efficient/slim_adam/layer_map_llama_paper.json')


def build_command(lr: str, root: Path, date: str) -> tuple[list[str], Path, str]:
    if lr not in LRS:
        raise ValueError(f'LR must be one of {LRS}')
    group = f'slimadam_paper_257M_1xC_{date}'
    name = f'slimadam_paper_257m_bf16_lr{lr}_wd1e-1_1xC_2gpu'
    metrics = root / group / name / 'metrics.jsonl'
    command = ['src/main.py', '--distributed-backend', 'nccl', '--experiment-name', name,
               '--seed', '0', '--data-seed', '1337', '--dataset', 'fineweb',
               '--datasets-dir', str(DATA), '--streaming', '--workers', '8',
               '--fineweb-replay-world-size', '1', '--fineweb-replay-layout', 'concat',
               '--model', 'llama', '--n-layer', '12', '--n-embd', '1024', '--n-head', '8',
               '--multiple-of', '256', '--sequence-length', '1024', '--dtype', 'bfloat16',
               '--dropout', '0', '--opt', 'slim_adam', '--lr', lr, '--weight-decay', '0.1',
               '--slim_adam_layer_map', str(LAYER_MAP),
               '--beta1', '0.9', '--beta2', '0.99', '--eps', '1e-8', '--grad-clip', '1',
               '--batch-size', '16', '--acc-steps', '8', '--eval-batch-size', '32',
               '--iterations', str(ITERATIONS), '--warmup-steps', '2000', '--scheduler', 'wsd',
               '--wsd-fract-decay', '0.1', '--wsd-final-lr-scale', '0', '--decay-type', 'cosine',
               '--latest-ckpt-interval', '0', '--permanent-ckpt-interval', '0', '--no-local-save',
               '--results-base-folder', str(root), '--metrics-jsonl', str(metrics),
               '--eval-interval', '500', '--eval-batches', '32', '--log-interval', '50',
               '--wandb', '--wandb-project', 'fp8-pretrain', '--wandb-group', group,
               '--wandb-tags', 'slim_adam', 'paper_map', '257M', '1xC', 'bf16', 'fp32_states', 'lr_tuning']
    return command, metrics, name


def audit_compression(command: list[str]) -> dict:
    import torch
    from config.base import parse_args
    from models.llama import Llama
    from optim.memory_efficient.slim_adam.slim_adam import get_rule_for_parameter

    args = parse_args(argparse.ArgumentParser(), command[1:], None)
    layer_map = json.loads(LAYER_MAP.read_text())
    expected = {'attention_query': (1,), 'attention_key': (1,),
                'attention_value': (0,), 'attention_output': (0,),
                'mlp_up': (0,), 'mlp_gate': (0,), 'mlp_down': (0,),
                'token_embedding': (1,), 'lm_head': (1,), 'norm': None}
    with torch.device('meta'):
        model = Llama(args)
    full = compressed = matrices = 0
    assignments = []
    for name, parameter in model.named_parameters():
        matches = [kind for kind, patterns in layer_map.items()
                   if any(pattern in name for pattern in patterns)]
        if len(matches) != 1:
            raise RuntimeError(f'Ambiguous or missing compression rule: {name}: {matches}')
        dims = get_rule_for_parameter(name, layer_map)
        if dims != expected[matches[0]] or (dims is None) != (parameter.ndim == 1):
            raise RuntimeError(f'Unexpected compression axes: {name}: {dims}')
        shape = list(parameter.shape)
        for dim in dims or ():
            shape[dim] = 1
        full += parameter.numel()
        compressed += math.prod(shape)
        matrices += int(name.startswith('transformer.h.') and parameter.ndim == 2)
        assignments.append(dict(name=name, weight_shape=list(parameter.shape),
                                compress_dims=dims, second_moment_shape=shape))
    if (full, compressed, matrices) != (257188864, 233728, 84):
        raise RuntimeError(f'Unexpected model or compression: {full}, {compressed}, {matrices}')
    return dict(parameters=full, second_moment_elements=compressed,
                hidden_matrices_compressed=matrices, assignments=assignments,
                layer_map_sha256=hashlib.sha256(LAYER_MAP.read_bytes()).hexdigest())


def check_data(root: Path) -> dict:
    metadata = json.loads((DATA / 'packed_metadata.json').read_text())
    if (metadata.get('format') != 'packed_fineweb_h200_v3' or metadata.get('world_size') != 2
            or metadata.get('batch_size') != 16 or metadata.get('iterations', 0) < ITERATIONS):
        raise RuntimeError('Packed FineWeb metadata does not match the reference campaign')
    for item in metadata['ranks']:
        for segment in item.get('segments', [item]):
            path = DATA / segment['file']
            if not path.is_file() or path.stat().st_size != segment['bytes']:
                raise RuntimeError(f'Missing or incomplete packed data: {path}')
    if shutil.disk_usage(root).free < 10**9:
        raise RuntimeError('Output filesystem needs at least 1GB free')
    return metadata


def verify_final(metrics: Path) -> dict:
    records = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    finals = [r for r in records if r.get('event') == 'validation' and r.get('iter') == ITERATIONS]
    if not finals:
        raise RuntimeError('No final validation at the full token budget')
    record = finals[-1]
    loss = record.get('final-val/loss')
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        raise RuntimeError('Missing or nonfinite native final validation loss')
    if record.get('consumed_tokens') != TOKENS:
        raise RuntimeError('Final token budget does not match the campaign')
    return record


def wait_json(path: Path) -> dict:
    deadline = time.monotonic() + 180
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(1)
    return json.loads(path.read_text())


def write_json(path: Path, record: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dry-run', action='store_true')
    options = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    sys.path.insert(0, str(Path('src').resolve()))
    sys.path.insert(0, str(Path.cwd()))
    lr = os.environ['LR']
    date = os.environ.get('CAMPAIGN_DATE', '20261023')
    root = Path(os.environ.get('RESULTS_DIR', f'/workspace-SR006.nfs2/dimativator/slimadam-paper-{date}'))
    command, metrics, name = build_command(lr, root, date)
    if options.dry_run:
        print(json.dumps(dict(command=command, metrics=str(metrics), tokens=TOKENS), indent=2))
        return
    import torch

    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
    world = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
    if world not in (1, 2) or (world == 1 and torch.cuda.device_count() != 2):
        raise RuntimeError('Each LR job requires exactly two GPUs / two MPI ranks')
    root.mkdir(parents=True, exist_ok=True)
    check_data(root)
    for directory in ('claims', 'rendezvous', 'logs', 'finals', 'wandb'):
        (root / directory).mkdir(exist_ok=True)
    log = root / 'logs' / f'{name}_rank{rank}.log'
    ready = root / 'rendezvous' / f'{name}.json'
    final = root / 'finals' / f'{name}.json'
    claim = root / 'claims' / f'{name}_rank{rank}.json'
    with claim.open('x') as output:
        json.dump(dict(lr=lr, command=command, rank=rank, submitted_utc=datetime.now(timezone.utc).isoformat()), output)
    if rank == 0:
        if metrics.exists() or ready.exists() or final.exists() or log.exists():
            raise RuntimeError(f'Refusing duplicate or overwrite: {name}')
        audit = audit_compression(command)
        audit.update(command=command, optimizer='slim_adam', lr=float(lr), weight_decay=0.1,
                     betas=[0.9, 0.99], seed=0, data_seed=1337, iterations=ITERATIONS,
                     tokens=TOKENS, global_batch=128, checkpoint_saving=False,
                     compute_dtype='bfloat16', parameter_dtype='float32', state_dtype='float32')
        audit['code_revision'] = os.environ['CODE_REVISION']
        write_json(root / 'claims' / f'{name}_config.json', audit)
        print('SLIMADAM_PAPER_CONFIG=' + json.dumps({k: v for k, v in audit.items() if k != 'assignments'}), flush=True)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(('8.8.8.8', 80))
            address = connection.getsockname()[0]
        write_json(ready, dict(master_addr=os.environ.get('MASTER_ADDR', address), master_port=29500))
    rendezvous = wait_json(ready)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PYTORCH_ALLOC_CONF='expandable_segments:True', WANDB_MODE='offline',
               WANDB_BASE_URL='https://wandb-radfan.ru', WANDB_ENTITY='andrey',
               WANDB_DIR=str(root / 'wandb'), WANDB_RUN_ID=f'{name}-{date}',
               TRITON_CACHE_DIR=f'/tmp/slimadam-paper-{lr}-{rank}-{os.getpid()}')
    if world == 2:
        env.update(RANK=str(rank), WORLD_SIZE='2', LOCAL_RANK=os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'),
                   MASTER_ADDR=rendezvous['master_addr'], MASTER_PORT=str(rendezvous['master_port']))
        launch = [sys.executable]
    else:
        launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    subprocess.run(['df', '-h', str(root), str(DATA)], check=True)
    subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free', '--format=csv'], check=True)
    subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv'], check=True)
    print(f'RUN_START lr={lr} rank={rank} log={log}', flush=True)
    with log.open('x') as output:
        process = subprocess.Popen(launch + command, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            output.write(line)
            output.flush()
            print(line, end='', flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, launch + command)
    if rank == 0:
        result = verify_final(metrics)
        result.update(lr=float(lr), weight_decay=0.1, betas=[0.9, 0.99], checkpoint_saving=False,
                      layer_map=str(LAYER_MAP), second_moment_elements=233728,
                      wandb_url=f'https://wandb-radfan.ru/andrey/fp8-pretrain/runs/{env["WANDB_RUN_ID"]}')
        write_json(final, result)
        print('RUN_VERIFIED=' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
