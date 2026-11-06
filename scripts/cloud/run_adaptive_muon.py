"""Cloud LR sweep and 500M trunk/decay entrypoint for Muon² and NorMuon."""
import json
import math
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time

ROOT = Path(os.environ.get('RESULTS_DIR', '/workspace-SR006.nfs3/dimativator/adaptive-muon-20261011'))
SOURCE_ROOT = Path(os.environ.get('SWEEP_RESULTS_DIR', str(ROOT)))
RECOVERY = os.environ.get('RECOVERY_RESULTS_DIR')
if RECOVERY:
    ROOT = Path(RECOVERY)
DATA = Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed')
LRS = ('1e-4', '5e-4', '1e-3', '2e-3')
OPTS = ('muon2', 'normuon')
RANK = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
WORLD = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
MODE = os.environ.get('MODE', 'smoke')
CAMPAIGN_DATE = os.environ.get('CAMPAIGN_DATE', '20261011')
TRAIN_WORLD = int(os.environ.get('CLOUD_TRAIN_WORLD_SIZE', '2'))
if TRAIN_WORLD not in (1, 2, 4) or (TRAIN_WORLD != 2 and MODE != 'matched'):
    raise ValueError('Only matched campaigns support one or four training GPUs')


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def wait_json(path: Path, timeout: int = 180) -> dict:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(1)
    return json.loads(path.read_text())


def final_metric(path: Path, target: int) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    finals = [r for r in records if r.get('event') == 'validation' and r.get('iter') == target]
    if not finals:
        raise RuntimeError(f'No final validation at {target}: {path}')
    result = finals[-1]
    loss = result.get('final-val/loss', result.get('val/loss'))
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        raise RuntimeError(f'Nonfinite final loss: {result}')
    if result['consumed_tokens'] != target * 128 * 1024:
        raise RuntimeError(f'Token budget mismatch: {result}')
    return {**result, 'val_loss': loss}


def train(opt: str, lr: str, size: int, phase: str, target: int, resume: Path | None = None, weight_decay: str = '1e-1') -> None:
    group = f'adaptive_muon_{size}M_{phase}_{CAMPAIGN_DATE}'
    if opt == 'soft_lion_muon':
        group = f'soft_lion_muon_{size}M_{phase}_{CAMPAIGN_DATE}'
    if opt == 'anvil2':
        group = f'anvil2_{size}M_{phase}_{CAMPAIGN_DATE}'
    name = f'{opt}_{size}m_bf16_lr{lr}_wd{weight_decay}_{phase}_{TRAIN_WORLD}gpu'
    if MODE == '257m_aux_lr':
        name += '_emb3e-1_head3e-3'
    if RECOVERY:
        name += '_storage_retry1'
    if opt == 'slim_normuon_lion':
        name += f'_p{int(os.environ["SLIM_LION_PERIOD"])}'
    decay_resume = resume is not None and phase == '1xC_decay'
    checkpoint_saving = (phase == '2xC' and resume is None
                         and os.environ.get('CLOUD_DISABLE_CHECKPOINTS') != '1')
    run_dir = ROOT / group / name
    metrics = run_dir / 'metrics.jsonl'
    done = ROOT / 'finals' / f'{name}.json'
    ready = ROOT / 'rendezvous' / f'{name}.json'
    if RANK == 0:
        if metrics.exists() or done.exists() or ready.exists():
            raise RuntimeError(f'Refusing duplicate or overwrite: {run_dir}')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(('8.8.8.8', 80))
            address = connection.getsockname()[0]
        atomic_json(ready, {'master_addr': os.environ.get('MASTER_ADDR', address), 'master_port': 29500})
    rendezvous = wait_json(ready)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PYTORCH_ALLOC_CONF='expandable_segments:True', WANDB_MODE=os.environ.get('WANDB_MODE', 'offline'),
               WANDB_BASE_URL='https://wandb-radfan.ru', WANDB_ENTITY='andrey',
               WANDB_DIR=str(ROOT / 'wandb'), WANDB_RUN_ID=f'adapt-{name}-{CAMPAIGN_DATE}',
               TRITON_CACHE_DIR=str(ROOT / 'cache' / f'triton-{opt}-{RANK}-{os.getpid()}') if MODE in ('500m_slim_normuon', 'matched') else f'/tmp/adaptive-muon-{opt}-{RANK}-{os.getpid()}')
    if WORLD > 1:
        if WORLD != TRAIN_WORLD:
            raise RuntimeError(f'Exactly {TRAIN_WORLD} MPI ranks required')
        env.update(RANK=str(RANK), WORLD_SIZE=str(TRAIN_WORLD),
                   LOCAL_RANK=os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'),
                   MASTER_ADDR=rendezvous['master_addr'], MASTER_PORT=str(rendezvous['master_port']))
        launch = [sys.executable]
    else:
        launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={TRAIN_WORLD}']
    layers, width, heads = (12, 1024, 8) if size == 257 else (18, 1280, 20)
    args = ['src/main.py', '--distributed-backend', 'nccl', '--experiment-name', name,
            '--seed', '0', '--data-seed', '1337', '--dataset', 'fineweb', '--datasets-dir', str(DATA),
            '--fineweb-replay-world-size', '2' if TRAIN_WORLD == 1 else '1', '--fineweb-replay-layout', 'concat',
            '--sequence-length', '1024', '--streaming', '--workers', '8',
            '--model', 'llama', '--n-layer', str(layers), '--n-embd', str(width),
            '--n-head', str(heads), '--multiple-of', '256', '--dtype', 'bfloat16', '--dropout', '0',
            '--opt', opt, '--lr', lr, '--weight-decay', weight_decay, '--momentum', '0.95',
            '--muon_beta2', '0.95', '--muon_ns_steps', '5', '--muon2-ns-steps', '5',
            '--beta1', '0.9', '--beta2', '0.95', '--eps', '1e-8', '--grad-clip', '1',
            '--batch-size', str(32 // TRAIN_WORLD), '--acc-steps', str(4 * TRAIN_WORLD), '--eval-batch-size', '32',
            '--iterations', str(target), '--scheduler', 'wsd', '--warmup-steps', '0' if decay_resume else '2000',
            '--wsd-fract-decay', '1.0' if decay_resume else '0.1', '--wsd-final-lr-scale', '0', '--decay-type', 'cosine',
            '--latest-ckpt-interval', '0', '--permanent-ckpt-interval', '0',
            '--results-base-folder', str(ROOT), '--metrics-jsonl', str(metrics),
            '--wandb', '--wandb-project', 'fp8-pretrain', '--wandb-group', group,
            '--wandb-tags', opt, f'{size}M', phase, 'bf16', 'fp32_states', 'lr_tuning',
            'anvil_maps6' if opt == 'anvil2' else 'ns5']
    smoke = MODE == 'smoke' or phase in ('aux_smoke', 'slim_smoke', 'matched_smoke')
    if MODE == 'matched' and opt != 'anvil2':
        args += ['--muon_adamw_lr_scale', str(0.001 / float(lr))]
    if opt == 'anvil2':
        args += ['--anvil2-engage-fraction', '0.5', '--anvil2-momentum-warmup-steps', '240',
                 '--anvil2-momentum-cooldown-steps', '50', '--model-parameter-dtype', 'float32']
    if opt == 'softmuon':
        args += ['--softmuon-aux-lr', '0.001', '--eps', '1e-10',
                 '--softmuon-sign-fraction', '0.9', '--softmuon-eps', '1e-4',
                 '--softmuon-newton-iters', '10']
    if opt == 'soft_lion_muon':
        args += ['--soft-lion-muon-aux-lr', '0.001', '--soft-lion-muon-period', '5',
                 '--soft-lion-muon-beta1', '0.95', '--soft-lion-muon-beta2', '0.99',
                 '--soft-lion-muon-start-fraction', '0.9', '--soft-lion-muon-eps', '1e-4',
                 '--soft-lion-muon-ns-steps', '6', '--soft-lion-muon-soft-ns-steps', '30']
    if MODE == '257m_aux_lr':
        args += ['--normuon-embedding-lr', '0.3', '--normuon-output-lr', '0.003']
    smoke_steps = 2 * int(os.environ['SLIM_LION_PERIOD']) if opt == 'slim_normuon_lion' else 2
    if opt == 'slim_normuon_lion':
        args += ['--slim-lion-period', os.environ['SLIM_LION_PERIOD'],
                 '--slim-lion-lr-ratio', '0.01', '--slim-lion-momentum', '0.99']
    if smoke:
        args += ['--early-stop-iteration', str(smoke_steps), '--eval-interval', str(smoke_steps), '--eval-batches', '1', '--log-interval', '1', '--no-local-save']
    else:
        args += ['--eval-interval', '500', '--eval-batches', '32', '--log-interval', '50']
        if checkpoint_saving:
            args += ['--inter-ckpts', '67911']
        else:
            args += ['--no-local-save']
        if resume:
            for filename in ('main.pt', 'worker_0.pt', 'worker_1.pt'):
                if not (resume / filename).is_file():
                    raise RuntimeError(f'Missing checkpoint file: {resume / filename}')
            args += ['--resume-from', str(resume)]
            if decay_resume:
                args += ['--decay-from-checkpoint']
    log = ROOT / 'logs' / f'{name}_rank{RANK}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'RUN_START optimizer={opt} lr={lr} size={size} phase={phase} rank={RANK} log={log}', flush=True)
    try:
        with log.open('x') as output:
            if MODE in ('500m_slim_normuon', 'matched'):
                process = subprocess.Popen(launch + args, env=env, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                last_report = 0.0
                last_disk_check = 0.0
                try:
                    while process.poll() is None:
                        if opt in ('softmuon', 'soft_lion_muon', 'anvil2') and time.monotonic() - last_disk_check >= 30:
                            from optim.disk_guard import DiskSpaceEmergency, available
                            last_disk_check = time.monotonic()
                            paths = [Path(p) for p in json.loads(env['CLOUD_DISK_GUARD_PATHS'])]
                            records = available(paths)
                            emergency = any(r['available_bytes'] <= 5_000_000_000 for r in records)
                            print('LAUNCHER_DISK_GUARD=' + json.dumps(dict(timestamp=time.time(),
                                  rank=RANK, paths=records, emergency=emergency)), flush=True)
                            if emergency:
                                print('КРИТИЧЕСКИ МАЛО МЕСТА НА CLOUD.RU. ОСТАНАВЛИВАЮ ОБУЧЕНИЕ.', flush=True)
                                raise DiskSpaceEmergency('Launcher storage reserve unavailable')
                        if RANK == 0 and time.monotonic() - last_report >= 60:
                            last_report = time.monotonic()
                            live = {'phase': phase, 'snapshot_time': time.time()}
                            try:
                                status = ROOT / 'disk_status.json'
                                if status.exists():
                                    live['disk'] = json.loads(status.read_text())
                                if metrics.exists():
                                    with metrics.open('rb') as stream:
                                        stream.seek(max(0, metrics.stat().st_size - 32768))
                                        lines = stream.read().splitlines()
                                    for line in reversed(lines):
                                        try:
                                            live['metric'] = json.loads(line)
                                            break
                                        except json.JSONDecodeError:
                                            continue
                            except (OSError, ValueError) as exc:
                                live['telemetry_error'] = str(exc)
                            print('LIVE_STATUS=' + json.dumps(live), flush=True)
                        time.sleep(2)
                    if process.returncode:
                        raise subprocess.CalledProcessError(process.returncode, launch + args)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            else:
                subprocess.run(launch + args, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        print('\n'.join(log.read_text().splitlines()[-80:]), flush=True)
        raise
    if RANK == 0:
        if smoke:
            result = final_metric(metrics, smoke_steps)
            result.update(smoke=True, optimizer=opt, lr=lr, size=size)
            print('\n'.join(log.read_text().splitlines()[-25:]), flush=True)
        else:
            result = final_metric(metrics, target)
            result.update(optimizer=opt, lr=lr, size=size, phase=phase, weight_decay=float(weight_decay),
                          ns_steps=5, checkpoint_saving=checkpoint_saving,
                          wandb_url=f'https://wandb-radfan.ru/andrey/fp8-pretrain/runs/{env["WANDB_RUN_ID"]}')
        if MODE in ('257m_slim_normuon', '500m_slim_normuon'):
            result.update(second_moment_source='post_ns_hidden_raw_grad_auxiliary',
                          compression_map='layer_map_llama_paper.json',
                          auxiliary_lr=float(lr), auxiliary_weight_decay=0.0,
                          matrix_momentum=0.95, matrix_beta2=0.95, adamw_betas=[0.9, 0.95],
                          weight_decay=float(weight_decay), ns_steps=5, checkpoint_saving=checkpoint_saving)
        if MODE == '257m_aux_lr':
            result.update(embedding_lr=0.3, output_lr=0.003, norm_lr=float(lr),
                          adamw_betas=[0.9, 0.95], matrix_momentum=0.95, matrix_beta2=0.95,
                          ns_steps=5, weight_decay=float(weight_decay), checkpoint_saving=False)
        if MODE == 'matched' and opt in ('slim_normuon_matched', 'slim_normuon_lion'):
            result.update(second_moment_source='post_ns_hidden_raw_grad_auxiliary',
                          compression_map='layer_map_llama_paper.json', second_moment_normalization=True,
                          compression_rule='Q_K_dim1_V_O_up_gate_down_dim0_embedding_head_dim1_norm_full')
        if MODE == 'matched' and opt not in ('softmuon', 'soft_lion_muon', 'anvil2'):
            result.update(update_rms=0.2, row_normalization=opt == 'normuon_matched',
                          ns_kernel='official_quintic_3.4445_-4.7750_2.0315',
                          auxiliary_lr=0.001, auxiliary_weight_decay=0.0,
                          adamw_betas=[0.9, 0.95], matrix_momentum=0.95, matrix_beta2=0.95,
                          ns_steps=5, weight_decay=float(weight_decay), checkpoint_saving=checkpoint_saving,
                          wandb_mode=env['WANDB_MODE'], physical_world_size=TRAIN_WORLD,
                          global_batch_size=128, microbatch_per_gpu=32 // TRAIN_WORLD,
                          accumulation_per_gpu=4)
        if opt == 'softmuon':
            result.update(algorithm='Algorithm3', auxiliary_lr=0.001,
                          auxiliary_weight_decay=float(weight_decay),
                          matrix_momentum=0.95, adamw_betas=[0.9, 0.95], adamw_eps=1e-10,
                          nesterov=False, temperature_spectrum='updated_momentum',
                          sign_iters=int(target * 0.9), transition_iters=target-int(target * 0.9),
                          update_rms=None, update_scale='original_shape_factor',
                          checkpoint_saving=False, physical_world_size=TRAIN_WORLD,
                          global_batch_size=128, microbatch_per_gpu=32 // TRAIN_WORLD,
                          accumulation_per_gpu=4, seed=0, data_seed=1337,
                          wandb_mode=env['WANDB_MODE'])
        if opt == 'anvil2':
            result.update(algorithm='ANVIL_II', auxiliary_lr=float(lr),
                          auxiliary_weight_decay=float(weight_decay),
                          adamw_betas=[0.9, 0.95], adamw_eps=1e-8,
                          auxiliary_update_every=1, cautious_weight_decay=True,
                          engage_fraction=0.5, engage_step=math.ceil(target * 0.5),
                          ns_steps=6, checkpoint_saving=False, physical_world_size=TRAIN_WORLD,
                          global_batch_size=128, seed=0, data_seed=1337,
                          wandb_mode=env['WANDB_MODE'])
        if opt == 'soft_lion_muon':
            result.update(algorithm='SoftLionMuon', period=5, lion_lr_ratio=0.01,
                          direction_beta=0.95, momentum_beta=0.99,
                          auxiliary_lr=0.001, auxiliary_weight_decay=float(weight_decay),
                          adamw_betas=[0.9, 0.95], adamw_eps=1e-8,
                          temperature_source='pre_update_lion_direction', separate_temperatures=True,
                          transition_start=int(target * 0.9), epsilon=1e-4,
                          ns_steps=6, soft_ns_steps=30, checkpoint_saving=False,
                          physical_world_size=TRAIN_WORLD, global_batch_size=128,
                          seed=0, data_seed=1337, wandb_mode=env['WANDB_MODE'])
        if opt == 'slim_normuon_lion':
            result.update(period=int(os.environ['SLIM_LION_PERIOD']), lion_lr_ratio=0.01,
                          lion_lr=0.01 * float(lr), direction_beta=0.95, matrix_momentum=0.99,
                          momentum_rule='shared_lion_pre_update_direction',
                          second_moment_clock='slim_steps_only', slim_update_rms=0.2,
                          update_rms=None, first_slim_step=int(os.environ['SLIM_LION_PERIOD']),
                          fineweb_replay_world_size=2 if TRAIN_WORLD == 1 else 1)
        atomic_json(done, result)
        print('RUN_VERIFIED=' + json.dumps(result, sort_keys=True), flush=True)
    wait_json(done)


def best_lr(opt: str) -> str:
    finals = []
    for lr in LRS:
        path = SOURCE_ROOT / 'finals' / f'{opt}_257m_bf16_lr{lr}_wd1e-1_1xC_2gpu.json'
        record = wait_json(path, timeout=1)
        if record.get('iter') != 39250:
            raise RuntimeError('Sweep not complete')
        finals.append(record)
    return min(finals, key=lambda r: r['val_loss'])['lr']


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[2])
    (ROOT / 'wandb').mkdir(parents=True, exist_ok=True)
    metadata = json.loads((DATA / 'packed_metadata.json').read_text())
    assert metadata['format'] == 'packed_fineweb_h200_v3'
    assert metadata['iterations'] >= 150914
    assert metadata['world_size'] == 2 and metadata['batch_size'] == 16
    if MODE == '257m_aux_lr':
        import shutil
        if shutil.disk_usage(ROOT).free < 10**9:
            raise RuntimeError('The auxiliary LR experiment needs at least1GB of free output storage')
    subprocess.run(['df', '-h', str(ROOT), str(DATA)], check=True)
    subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free', '--format=csv'], check=True)
    subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv'], check=True)
    if MODE == 'smoke':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_adaptive_muon.py', '-v'],
                       env={**os.environ, 'PYTHONPATH': 'src:.'}, check=True)
        for opt in OPTS:
            train(opt, '1e-3', 257, 'smoke', 39250)
    elif MODE == 'sweep':
        lr = os.environ['LR']
        assert lr in LRS
        for opt in OPTS:
            train(opt, lr, 257, '1xC', 39250)
    elif MODE == '257m_slim_normuon':
        if RECOVERY:
            raise RuntimeError('SlimNorMuon must start from scratch')
        lr = os.environ['LR']
        if lr not in LRS:
            raise ValueError('Unrequested LR')
        train('slim_normuon', lr, 257, 'slim_smoke', 39250, weight_decay='1e-1')
        train('slim_normuon', lr, 257, '1xC', 39250, weight_decay='1e-1')
    elif MODE == '500m_slim_normuon':
        if RECOVERY:
            raise RuntimeError('This campaign starts from scratch')
        train('slim_normuon', '1e-3', 500, 'slim_smoke', 150914, weight_decay='1e-4')
        train('slim_normuon', '1e-3', 500, '2xC', 150914, weight_decay='1e-4')
        checkpoint = ROOT / f'adaptive_muon_500M_2xC_{CAMPAIGN_DATE}' / 'slim_normuon_500m_bf16_lr1e-3_wd1e-4_2xC_2gpu' / 'ckpts' / '67911'
        train('slim_normuon', '1e-3', 500, '1xC_decay', 75457, resume=checkpoint, weight_decay='1e-4')
    elif MODE == '257m_aux_lr':
        if RECOVERY:
            raise RuntimeError('The auxiliary LR experiment must start from scratch')
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_adaptive_muon.py', '-v'],
                       env={**os.environ, 'PYTHONPATH': 'src:.'}, check=True)
        train('normuon', '1e-3', 257, 'aux_smoke', 39250, weight_decay='1e-1')
        train('normuon', '1e-3', 257, '1xC', 39250, weight_decay='1e-1')
    elif MODE == '500m_1xc':
        if RECOVERY:
            raise RuntimeError('The corrected1xC run must start from scratch')
        opt = os.environ['OPTIMIZER']
        assert opt in OPTS
        winners = {optimizer: best_lr(optimizer) for optimizer in OPTS}
        lr = winners[opt]
        if RANK == 0:
            atomic_json(ROOT / f'best_lr_{opt}.json', {'optimizer': opt, 'lr': lr})
        train(opt, lr, 500, '1xC', 75457, weight_decay='1e-4')
    elif MODE == '500m':
        opt = os.environ['OPTIMIZER']
        assert opt in OPTS
        # Require all eight sweep completions before either 500M launch.
        winners = {optimizer: best_lr(optimizer) for optimizer in OPTS}
        lr = winners[opt]
        if RANK == 0:
            atomic_json(ROOT / f'best_lr_{opt}.json', {'optimizer': opt, 'lr': lr})
        source = SOURCE_ROOT / 'adaptive_muon_500M_2xC_20261011' / f'{opt}_500m_bf16_lr{lr}_wd1e-1_2xC_2gpu' / 'ckpts' / '67911'
        if RECOVERY:
            import shutil
            if shutil.disk_usage(ROOT).free < 2 * 10**9:
                raise RuntimeError('Recovery output filesystem needs at least2GB free')
        train(opt, lr, 500, '2xC', 150914, resume=source if RECOVERY else None)
        train(opt, lr, 500, '1xC_decay', 75457, resume=source)
    else:
        raise ValueError(MODE)


if __name__ == '__main__':
    main()
