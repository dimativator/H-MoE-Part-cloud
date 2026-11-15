"""Short one-GPU diagnosis of full step, optimizer, and Newton-Schulz time."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import benchmark_lion_ttt_single as benchmark
import run_lionmuon_time_target as common

ROOT = common.ROOT / 'phase_profile'
DATA_REVISION = '3e7604f1210d42eb1c83f34af24b9f542fee81c8'


def export(run, status: str, **fields: object) -> None:
    value = dict(status=status,timestamp=time.time(),code_revision=os.environ['CODE_REVISION'],
                 measurement_gpus=1,microbatch=32,accumulation=4,global_batch=128,**fields)
    (ROOT/'status.json').write_text(json.dumps(value,indent=2))
    run.summary['phase_profile'] = value
    common.emit('PHASE_STATUS',value)


def profile(optimizer: str, run, results: dict) -> dict:
    name='profile_'+optimizer
    log=ROOT/(name+'.log');result=ROOT/(name+'.json')
    if log.exists() or result.exists():raise RuntimeError('Duplicate profiling refused')
    argv=benchmark.command(name,optimizer,32,True)
    argv[argv.index('--early-stop-iteration')+1]='90'
    argv[argv.index('--metrics-jsonl')+1]=str(ROOT/name/'metrics.jsonl')
    env={**os.environ,'WANDB_MODE':'disabled','LION_TTT_BENCHMARK':'0',
         'LION_TTT_PHASE_PROFILE':str(result),'DATA_PROOF_REVISION':DATA_REVISION}
    for field in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):env.pop(field,None)
    with log.open('x') as output:
        child=subprocess.Popen(argv,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        last=0.0
        try:
            while child.poll() is None:
                if time.monotonic()-last>=30:
                    last=time.monotonic();space=common.space()
                    export(run,'running',current_optimizer=optimizer,child_pid=child.pid,results=results,space=space)
                time.sleep(1)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    if child.returncode:raise RuntimeError('Profile child failed: '+log.read_text()[-6000:])
    if 'EARLY_STOP_ITERATION=90' not in log.read_text():raise RuntimeError('Missing exact profiling completion')
    value=json.loads(result.read_text())
    prior=json.loads((common.ROOT/'single_gpu_timing/timing_plan.json').read_text())
    if prior['code_revision']!=DATA_REVISION:raise RuntimeError('Uninstrumented comparison provenance mismatch')
    reference_ms=prior['timing'][optimizer]['seconds_per_step']*1000
    value['prior_uninstrumented_wall_ms']=reference_ms
    value['profiled_wall_delta_percent']=100*(value['mean_wall_ms']/reference_ms-1)
    return value


def main() -> None:
    signal.signal(signal.SIGTERM,lambda s,f:sys.exit(128+s))
    signal.signal(signal.SIGINT,lambda s,f:sys.exit(128+s))
    common.setup();ROOT.mkdir(exist_ok=True)
    if os.environ.get('CLOUD_DISABLE_CHECKPOINTS')!='1':raise RuntimeError('Checkpoints must be disabled')
    import wandb
    stage=os.environ.get('PHASE_STAGE','profile')
    run=wandb.init(entity='andrey',project='fp8-pretrain',id='lion-ttt-phase-'+stage+'-20261115',
        name='LionMuon phase diagnosis '+stage,dir=str(common.ROOT/'wandb'),resume='never',
        settings=wandb.Settings(init_timeout=60))
    try:
        if stage=='preflight':
            subprocess.run([sys.executable,'-m','py_compile','src/optim/base.py','src/optim/ttt_phase_profile.py'],check=True)
            import torch
            export(run,'preflight_complete',space=common.space(),torch=torch.__version__);return
        import torch
        if torch.cuda.device_count()!=1 or int(os.environ.get('OMPI_COMM_WORLD_SIZE','1'))!=1:
            raise RuntimeError('Exactly one allocated GPU required')
        results={}
        for optimizer in ('muon_reference','lion_muon'):
            results[optimizer]=profile(optimizer,run,results)
            export(run,'running',results=results,space=common.space())
        export(run,'complete',results=results,space=common.space())
    except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as exc:
        export(run,'failed',error=str(exc));raise
    finally:run.finish()


if __name__=='__main__':main()
