"""Authorized independent4GPU full runs using one-GPU batch32 timing."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_lionmuon_time_target as common

BENCHMARK_REVISION='933186b66a98aef840dc8a561438f8e59d5e90d7'
ROOT=common.ROOT/'full_batch32'
ORIGINAL_ARGUMENTS=common.arguments


def arguments(name:str,optimizer:str,target:int,decay:int,benchmark:bool)->list[str]:
    args=ORIGINAL_ARGUMENTS(name,optimizer,target,decay,benchmark)
    args[0]='scripts/cloud/lion_ttt_four_gpu_train.py'
    args[args.index('--batch-size')+1]='32'
    args[args.index('--acc-steps')+1]='4'  # DDP converts4global accumulation to1per rank.
    return args


def plan()->dict:
    value=json.loads((common.ROOT/'batch32_timing/status.json').read_text())
    if value['status']!='complete' or value['code_revision']!=BENCHMARK_REVISION:
        raise RuntimeError('Completed matching batch32 benchmark required')
    if (value['measurement_gpus'],value['microbatch'],value['accumulation'],value['global_batch'])!=(1,32,1,32):
        raise RuntimeError('Wrong measurement protocol')
    muon,lion=(value['timing'][key]['seconds_per_step'] for key in ('muon_reference','lion_muon'))
    if not all(math.isfinite(t) and t>0 for t in (muon,lion)):raise RuntimeError('Invalid timings')
    targets={}
    for trial,baseline in common.BASELINES.items():
        target=math.floor(baseline*muon/(1.05*lion))
        if target<=baseline:raise RuntimeError('Stable extension infeasible')
        targets[trial]=dict(baseline_steps=baseline,target_steps=target,decay_steps={'1xc':7546,'2xc':15091}[trial],
            extra_stable_steps=target-baseline,consumed_tokens=target*131072,
            muon_seconds=baseline*muon,lion_seconds=target*lion,predicted_speedup=baseline*muon/(target*lion))
    expected=json.loads((Path(__file__).with_name('lion_ttt_batch32_compute_hashes.json')).read_text())
    for path,digest in expected.items():
        if hashlib.sha256((common.REPO/path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Measured computational source changed: '+path)
    return dict(code_revision=os.environ['CODE_REVISION'],benchmark_revision=BENCHMARK_REVISION,
        measurement_gpus=1,measurement_global_batch=32,training_gpus=4,training_global_batch=128,
        per_gpu_microbatch=32,per_gpu_accumulation=1,warmup_steps=2000,targets=targets,timing=value['timing'],
        compute_sha256=expected,method='SingleGPU complete training step MB32acc1; time axis is equivalent singleGPU time, excludes validation/setup.')


def main()->None:
    signal.signal(signal.SIGTERM,lambda s,f:sys.exit(128+s));signal.signal(signal.SIGINT,lambda s,f:sys.exit(128+s))
    metadata=common.setup();ROOT.mkdir(exist_ok=True)
    if os.environ.get('CLOUD_DISABLE_CHECKPOINTS')!='1':raise RuntimeError('No checkpoints required')
    value=plan()
    if metadata['iterations']<max(t['target_steps'] for t in value['targets'].values()):
        raise RuntimeError('Data horizon insufficient')
    stage=os.environ['FULL32_STAGE']
    if stage in ('preflight','inspect'):
        import wandb
        run=wandb.init(entity='andrey',project='fp8-pretrain',id=os.environ.get('FULL32_WANDB_RUN_ID','lion-ttt-full32-'+stage+'-20261115'),
            dir=str(common.ROOT/'wandb'),resume='never',settings=wandb.Settings(init_timeout=60))
        try:
            if stage=='preflight':
                subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_lion_ttt_full32.py','-v'],check=True)
                (ROOT/'timing_plan.json').write_text(json.dumps(value,indent=2))
                run.summary['timing_plan']=value
                import torch
                from types import SimpleNamespace
                import lion_ttt_four_gpu_train as adapter
                from data import fineweb_packed as packed
                hash_cache={};last_guard=time.monotonic()
                def guarded_hash(path:Path)->str:
                    nonlocal last_guard
                    stat=path.stat();identity=(str(path),stat.st_size,stat.st_mtime_ns)
                    if identity in hash_cache:return hash_cache[identity]
                    digest=hashlib.sha256()
                    with path.open('rb') as stream:
                        while chunk:=stream.read(16*1024*1024):
                            digest.update(chunk)
                            if time.monotonic()-last_guard>=30:
                                last_guard=time.monotonic()
                                run.summary['full32_status']=dict(status='preflight_running',timestamp=time.time(),space=common.space())
                    after=path.stat()
                    if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns):raise RuntimeError('Data changed during hash')
                    hash_cache[identity]=digest.hexdigest()
                    return digest.hexdigest()
                packed._sha256_file=guarded_hash
                adapter.ORIGINAL_BUILD=packed.build_packed_fineweb_readers
                args=SimpleNamespace(datasets_dir=str(common.DATA),fineweb_replay_world_size=1,fineweb_replay_layout='concat',
                    batch_size=32,acc_steps=1,sequence_length=1024,eval_batch_size=32)
                proof=[]
                for rank in range(4):
                    readers=adapter.build_readers(args,rank=rank,world_size=4)
                    reader=readers['train']
                    for step in (0,value['targets']['2xc']['target_steps']-1):
                        reader.set_step(step);x,y=reader.sample_batch()
                        if tuple(x.shape)!=(32,1024) or tuple(y.shape)!=(32,1024):raise RuntimeError('Rebatch shape mismatch')
                        proof.append(dict(rank=rank,step=step,sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),source_step=reader.source.step))
                run.summary['rebatch_data_proof']=proof
            curves={name:common.records(name) for name in ('1xc','2xc')}
            finals={p.stem:json.loads(p.read_text()) for p in (common.ROOT/'finals').glob('*.json')}
            checkpoints=[str(p) for p in common.ROOT.rglob('*') if p.is_file() and p.suffix in ('.pt','.pth','.ckpt')]
            disk={p.stem:json.loads(p.read_text()) for p in common.ROOT.glob('launcher_disk_*.json')}
            bench_logs={opt:(common.ROOT/'batch32_timing'/(opt+'.log')).read_text()[-16000:] for opt in ('muon_reference','lion_muon')}
            result=dict(status=stage+'_complete',timestamp=time.time(),code_revision=os.environ['CODE_REVISION'],space=common.space(),
                curves={n:dict(validation=[r for r in rows if r['event']=='validation'],latest=rows[-3:]) for n,rows in curves.items()},
                finals=finals,checkpoint_paths=checkpoints,worker_disk=disk,benchmark_log_tails=bench_logs)
            run.summary['full32_status']=result;common.emit('FULL32_STATUS',result)
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
            run.summary['full32_status']=dict(status='failed',error=str(exc),timestamp=time.time());raise
        finally:run.finish()
        return
    if stage!='train':raise RuntimeError('Unsupported full32 stage')
    if int(os.environ.get('OMPI_COMM_WORLD_SIZE','1'))!=4:raise RuntimeError('Exactly4GPUs required')
    trial=os.environ['TRIAL'];target=value['targets'][trial]
    common.arguments=arguments
    os.environ['LION_TTT_MASTER_PORT']='29531' if trial=='1xc' else '29532'
    common.emit('FULL32_PLAN',value)
    common.run(trial,'lion_muon',target['target_steps'],target['decay_steps'],False)


if __name__=='__main__':main()
