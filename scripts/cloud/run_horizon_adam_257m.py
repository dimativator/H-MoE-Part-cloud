"""One-GPU HorizonAdam LR campaign with smoke gates and durable recovery outputs."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

LRS = ('1e-4', '5e-4', '1e-3', '2e-3')
GROUP = 'horizon_adam_257m_lr_20261014'
TARGET = 39250
TOKENS_PER_STEP = 128 * 1024
ROOT = Path(os.environ.get('RESULTS_DIR', '/workspace-SR006.nfs1/dimativator/horizon-adam-257m-lr-20261014'))
DATA = Path(os.environ.get('DATASETS_DIR', '/workspace-SR006.nfs3/dimativator/fineweb-h200-packed'))
REPO = Path(__file__).resolve().parents[2]
SCIENCE = dict(optimizer='horizon_adam', parameters='fp32', activations='bf16', layers=12,
               width=1024, heads=8, iterations=TARGET, batch_size=16, acc_steps=8,
               sequence_length=1024, seed=0, data_seed=1337, weight_decay=0.1,
               beta1=0.9, beta2=0.99, eps=1e-8, scheduler='wsd', warmup=2000,
               final_cosine_fraction=0.1, horizon=32, bits=8, block_size=256,
               chunk_size=65536, probes=64, min_size=4096, rounding_seed=0,
               replay_world_size=2, replay_layout='serial')


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def data_report() -> dict:
    metadata = json.loads((DATA / 'packed_metadata.json').read_text())
    expected = dict(format='packed_fineweb_h200_v3', world_size=2, batch_size=16,
                    sequence_length=1024, block_tokens=1025)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f'Dataset {key}: expected {value!r}, got {metadata.get(key)!r}')
    if metadata['iterations'] < TARGET:
        raise RuntimeError('Packed dataset is shorter than training horizon')
    files = []
    for rank in metadata['ranks']:
        segments = rank.get('segments', [rank])
        if sum(int(s['blocks']) for s in segments) != int(rank['blocks']):
            raise RuntimeError('Dataset segment blocks do not match rank metadata')
        for segment in segments:
            path = DATA / segment['file']
            actual = path.stat().st_size
            if actual != int(segment['bytes']) or actual != int(segment['blocks']) * 1025 * 2:
                raise RuntimeError(f'Dataset file size mismatch: {path}')
            files.append(dict(file=str(path), bytes=actual, sha256=segment['sha256']))
    validation = metadata['validation']
    path = DATA / validation['file']
    if path.stat().st_size != int(validation['bytes']):
        raise RuntimeError(f'Validation file size mismatch: {path}')
    files.append(dict(file=str(path), bytes=path.stat().st_size, sha256=validation['sha256']))
    # The normal packed reader verifies fingerprints and SHA-256 before training.
    return dict(path=str(DATA), metadata=metadata, files=files)


def storage_report(require_space: bool = True) -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(ROOT)
    # This 257M model plus packed optimizer state is about 1.31 GB. Reserve
    # 1.5 GB per run and another 2 GB for logs, metadata, and serialization.
    # Checkpoints already occupying this filesystem must not be reserved twice.
    checkpoint_bytes_per_run = 1_500_000_000
    checkpoint_bytes = {}
    for lr in LRS:
        attempts = []
        for directory in (ROOT / GROUP).glob(f'horizon_adam_257m_b8_h32_lr{lr}_wd1e-1_s0_attempt*'):
            manifest = directory / 'campaign.json'
            if manifest.is_file():
                attempts.append((json.loads(manifest.read_text())['attempt'], directory))
        latest = max(attempts, key=lambda item: item[0])[1] if attempts else None
        paths = [latest / 'ckpts' / 'latest' / filename for filename in ('main.pt', 'worker_0.pt')] if latest else []
        checkpoint_bytes[lr] = sum(path.stat().st_size for path in paths if path.is_file())
    required = 2_000_000_000 + sum(max(0, checkpoint_bytes_per_run - size) for size in checkpoint_bytes.values())
    result = dict(path=str(ROOT), total_bytes=disk.total, used_bytes=disk.used,
                  free_bytes=disk.free, required_free_bytes=required,
                  checkpoint_bytes_per_run=checkpoint_bytes_per_run,
                  existing_checkpoint_bytes=checkpoint_bytes)
    if require_space and disk.free < result['required_free_bytes']:
        raise RuntimeError(f'Insufficient space for remaining HorizonAdam checkpoints: {result}')
    return result


def storage_inventory() -> None:
    locations = [Path('/home/jovyan'), Path('/tmp'), *sorted(Path('/').glob('workspace-*'))]
    rows = []
    for location in locations:
        if not location.is_dir():
            continue
        disk = shutil.disk_usage(location)
        rows.append(dict(path=str(location), total_bytes=disk.total,
                         used_bytes=disk.used, free_bytes=disk.free))
    print('HORIZON_STORAGE_JSON=' + json.dumps(rows), flush=True)


def preflight() -> dict:
    report = dict(timestamp=time.time(), data=data_report(), storage=storage_report(),
                  group=GROUP, science=SCIENCE, lrs=LRS)
    print('HORIZON_PREFLIGHT_JSON=' + json.dumps(report, sort_keys=True), flush=True)
    return report


def gpu_smoke(lr: str) -> dict:
    # Run this before creating a training CUDA context, for an allocation audit.
    for query in ('--query-gpu=index,uuid,name,memory.total,memory.used,memory.free',
                  '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory'):
        subprocess.run(['nvidia-smi', query, '--format=csv'], check=True)
    import torch
    sys.path.insert(0, str(REPO / 'src'))
    from optim.horizon_adam import HorizonAdam
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Campaign requires exactly one visible CUDA GPU')
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(65536, device='cuda', dtype=torch.float32))
    optimizer = HorizonAdam([parameter], lr=float(lr), betas=(0.9, 0.99), weight_decay=0.1,
                            horizon=2, total_bits=8, min_size=4096)
    for _ in range(5):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        if not torch.isfinite(parameter).all():
            raise RuntimeError('GPU codec smoke produced nonfinite parameters')
    state = optimizer.state[parameter]
    if state['codec_updates'] < 2 or state['packed'].dtype != torch.uint8:
        raise RuntimeError('GPU smoke did not exercise codec refresh and packed state')
    for value in state.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError('GPU smoke produced nonfinite optimizer state')
    clone = torch.nn.Parameter(parameter.detach().clone())
    restored = HorizonAdam([clone], horizon=2)
    restored.load_state_dict(optimizer.state_dict())
    restored_state = restored.state[clone]
    if restored_state['packed'].dtype != torch.uint8 or not torch.equal(restored_state['packed'], state['packed']):
        raise RuntimeError('GPU packed-state checkpoint roundtrip failed')
    report = dict(device=torch.cuda.get_device_name(0), codec_updates=state['codec_updates'],
                  allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
    print('HORIZON_GPU_SMOKE_JSON=' + json.dumps(report), flush=True)
    del restored, optimizer, parameter, clone, state, restored_state
    torch.cuda.empty_cache()
    return report


def verify_metric(path: Path, step: int, final: bool = False) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    candidates = [r for r in records if r.get('event') == 'validation' and r.get('iter') == step]
    if not candidates:
        raise RuntimeError(f'Missing validation at step {step}: {path}')
    result = candidates[-1]
    key = 'final-val/loss' if final else 'val/loss'
    loss = result.get(key)
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        raise RuntimeError(f'Nonfinite or missing {key}: {result}')
    if result.get('consumed_tokens') != step * TOKENS_PER_STEP:
        raise RuntimeError(f'Token budget mismatch: {result}')
    return {**result, 'val_loss': loss}


def validate_resume(checkpoint: Path, lr: str) -> dict:
    """Reject partial writes and model/data checkpoints from different steps."""
    import torch
    for filename in ('main.pt', 'worker_0.pt'):
        if not (checkpoint / filename).is_file():
            raise RuntimeError(f'Missing recovery file: {checkpoint / filename}')
    source = json.loads((checkpoint.parent.parent / 'campaign.json').read_text())
    if source['science'] != SCIENCE or source['lr'] != lr:
        raise RuntimeError('Recovery checkpoint scientific configuration differs from this run')
    model = torch.load(checkpoint / 'main.pt', map_location='cpu', weights_only=False)
    worker = torch.load(checkpoint / 'worker_0.pt', map_location='cpu', weights_only=False)
    iteration = model['itr']
    reader = worker['train_reader_state']
    if not isinstance(iteration, int) or not 0 < iteration < TARGET:
        raise RuntimeError(f'Invalid recovery iteration: {iteration}')
    expected = dict(reader_type='fineweb_serial_replay_train_reader_v1', batch_size=16,
                    sequence_length=1024, source_world_size=2, step=iteration * 8)
    if any(reader.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f'Model/data checkpoint mismatch at iteration {iteration}: {reader}')
    sources = reader.get('source_states', [])
    if len(sources) != 2 or any(source.get('step') != iteration * 4 or source.get('rank') != rank
                                for rank, source in enumerate(sources)):
        raise RuntimeError('Recovery source-reader positions do not match the model checkpoint')
    if model.get('scheduler') is None or not model.get('optimizer'):
        raise RuntimeError('Recovery requires optimizer and scheduler state')
    if any(state.get('codec_version') != 1 for state in model['optimizer']['state'].values() if state):
        raise RuntimeError('Recovery requires HorizonAdam codec state')
    if any(key not in worker for key in ('rng_python', 'rng_np', 'rng_torch_cpu', 'rng_torch_gpu')):
        raise RuntimeError('Recovery requires complete RNG state')
    report = dict(path=str(checkpoint), iteration=iteration, reader_microstep=reader['step'])
    print('HORIZON_RECOVERY_VERIFIED_JSON=' + json.dumps(report), flush=True)
    return report


def train_args(name: str, lr: str, metrics: Path, smoke: bool, resume: Path | None) -> list[str]:
    args = ['src/main.py', '--distributed-backend', 'nccl', '--experiment-name', name,
            '--seed', '0', '--data-seed', '1337', '--dataset', 'fineweb', '--datasets-dir', str(DATA),
            '--fineweb-replay-world-size', '2', '--fineweb-replay-layout', 'serial',
            '--sequence-length', '1024', '--streaming', '--workers', '8',
            '--model', 'llama', '--n-layer', '12', '--n-embd', '1024', '--n-head', '8',
            '--multiple-of', '256', '--dtype', 'bfloat16', '--dropout', '0',
            '--opt', 'horizon_adam', '--lr', lr, '--weight-decay', '1e-1',
            '--beta1', '0.9', '--beta2', '0.99', '--eps', '1e-8', '--grad-clip', '1',
            '--horizon-adam-horizon', '32', '--horizon-adam-bits', '8',
            '--horizon-adam-block-size', '256', '--horizon-adam-chunk-size', '65536',
            '--horizon-adam-probes', '64', '--horizon-adam-min-size', '4096', '--horizon-adam-seed', '0',
            '--batch-size', '16', '--acc-steps', '8', '--eval-batch-size', '32',
            '--iterations', str(TARGET), '--scheduler', 'wsd', '--warmup-steps', '2000',
            '--wsd-fract-decay', '0.1', '--wsd-final-lr-scale', '0', '--decay-type', 'cosine',
            '--latest-ckpt-interval', '0' if smoke else '2000', '--permanent-ckpt-interval', '0',
            '--results-base-folder', str(ROOT), '--metrics-jsonl', str(metrics),
            '--wandb', '--wandb-project', 'fp8-pretrain', '--wandb-group', GROUP,
            '--wandb-tags', 'horizon_adam', '257M', 'bf16', 'fp32_parameters', 'lr_tuning', 'b8', 'h32']
    if smoke:
        args += ['--early-stop-iteration', '2', '--eval-interval', '2', '--eval-batches', '1',
                 '--log-interval', '1', '--no-local-save']
    else:
        args += ['--eval-interval', '500', '--eval-batches', '32', '--log-interval', '50']
        if resume is not None:
            args += ['--resume-from', str(resume)]
    return args


def run_training(name: str, lr: str, metrics: Path, log: Path, smoke: bool,
                 resume: Path | None = None) -> dict:
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PYTORCH_ALLOC_CONF='expandable_segments:True', WANDB_MODE='offline',
               WANDB_DIR=str(ROOT / 'wandb'), WANDB_RUN_ID=name,
               TRITON_CACHE_DIR=f'/tmp/horizon-adam-{os.getpid()}')
    (ROOT / 'wandb').mkdir(parents=True, exist_ok=True)
    launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1']
    command = launch + train_args(name, lr, metrics, smoke, resume)
    print('HORIZON_TRAIN_START_JSON=' + json.dumps(dict(name=name, command=command, log=str(log))), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open('a' if resume is not None else 'x') as output:
            subprocess.run(command, cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
        return verify_metric(metrics, 2 if smoke else TARGET, final=not smoke)
    except Exception:
        if log.exists():
            with log.open() as source:
                from collections import deque
                print(''.join(deque(source, maxlen=80)), flush=True)
        raise


def campaign() -> None:
    if int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1')) != 1 or int(os.environ.get('OMPI_COMM_WORLD_RANK', '0')) != 0:
        raise RuntimeError('One MPI rank required per campaign job')
    lr = os.environ['LR']
    if lr not in LRS:
        raise ValueError(f'LR must be one of {LRS}')
    attempt = int(os.environ.get('ATTEMPT', '0'))
    if attempt < 0:
        raise ValueError('ATTEMPT must be nonnegative')
    name = f'horizon_adam_257m_b8_h32_lr{lr}_wd1e-1_s0_attempt{attempt}'
    run_dir = ROOT / GROUP / name
    resume = Path(os.environ['RESUME_FROM']).resolve() if os.environ.get('RESUME_FROM') else None
    if run_dir.exists():
        raise RuntimeError(f'Refusing duplicate/overwrite: {run_dir}. Always use a new ATTEMPT, including recovery.')
    if resume is not None:
        validate_resume(resume, lr)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(name=name, group=GROUP, lr=lr, attempt=attempt, science=SCIENCE,
                    resume_from=str(resume) if resume else None, timestamp=time.time())
    atomic_json(run_dir / 'campaign.json', manifest)
    status = run_dir / 'status.json'
    try:
        atomic_json(status, dict(phase='preflight', timestamp=time.time()))
        preflight()
        atomic_json(status, dict(phase='gpu_smoke', timestamp=time.time()))
        gpu_smoke(lr)
        smoke_name = name + f'_smoke_{int(time.time())}'
        atomic_json(status, dict(phase='model_smoke', timestamp=time.time()))
        smoke = run_training(smoke_name, lr, run_dir / f'{smoke_name}.jsonl',
                             run_dir / f'{smoke_name}.log', smoke=True)
        atomic_json(run_dir / 'smoke.json', smoke)
        atomic_json(status, dict(phase='training', timestamp=time.time()))
        result = run_training(name, lr, run_dir / 'metrics.jsonl', run_dir / 'train.log', smoke=False, resume=resume)
        result.update(manifest)
        atomic_json(run_dir / 'done.json', result)
        atomic_json(status, dict(phase='complete', timestamp=time.time()))
        print('HORIZON_RUN_VERIFIED_JSON=' + json.dumps(result, sort_keys=True), flush=True)
    except Exception as error:
        failure = dict(timestamp=time.time(), exception=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc())
        atomic_json(run_dir / 'failure.json', failure)
        atomic_json(status, dict(phase='failed', **failure))
        print('HORIZON_RUN_FAILURE_JSON=' + json.dumps(failure, sort_keys=True), flush=True)
        raise


def main() -> None:
    os.chdir(REPO)
    mode = os.environ.get('MODE', 'preflight')
    if mode == 'preflight':
        preflight()
    elif mode == 'storage':
        storage_inventory()
    elif mode == 'campaign':
        campaign()
    elif mode == 'inspect':
        from inspect_horizon_adam_257m import main as inspect_main
        inspect_main()
    else:
        raise ValueError(f'Unknown MODE: {mode}')


if __name__ == '__main__':
    main()
