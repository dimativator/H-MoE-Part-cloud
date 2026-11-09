"""One-GPU Efficient-Training Muon/AdamW profiling; checkpoints forbidden."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from profile_storage import snapshot

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('/home/jovyan/efficient-muon-profile-20261109')
DATA = Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed')
CHILD = None

def guard():
    record = snapshot()
    bad = [r for r in record['records'] if r['writable_by_campaign'] and r.get('available_bytes', 0) <= 5_500_000_000]
    record['status'] = 'stop' if bad else 'ok'
    print('PROFILE_STORAGE=' + json.dumps(record), flush=True)
    if bad:
        if CHILD is not None and CHILD.poll() is None:
            os.killpg(CHILD.pid, signal.SIGTERM)
            try: CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(CHILD.pid, signal.SIGKILL)
        raise RuntimeError('CRITICAL CLOUD.RU STORAGE: TRACKED PROFILE STOPPED; NO CHECKPOINTS TO DELETE')
    if OUT.exists():
        with (OUT/'disk_guard.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')


def main():
    global CHILD
    parser = argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    guard();OUT.mkdir(exist_ok=True)
    metadata=json.loads((DATA/'packed_metadata.json').read_text());print('DATA='+json.dumps(metadata)[:1200],flush=True)
    os.chdir(ROOT)
    os.environ.update(WANDB_MODE='disabled',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
        TRITON_CACHE_DIR='/tmp/efficient-muon-triton',TORCHINDUCTOR_CACHE_DIR='/tmp/efficient-muon-inductor',CUDA_CACHE_PATH='/tmp/efficient-muon-cuda')
    if args.preflight:
        p=subprocess.run([sys.executable,'src/main.py','--help'],capture_output=True,text=True)
        print('IMPORT_CHECK='+str(p.returncode),flush=True)
        if p.returncode: print(p.stderr[-5000:],flush=True);raise RuntimeError('Training imports failed')
        print('PROFILE_PREFLIGHT_OK',flush=True);return
    sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
    from benchmark_peak_memory import command_for
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    results=[]
    for method in ['Muon','AdamW']:
        for mb in [1,32]:
            guard();name=f'{method}_fp8_mb{mb}';profile=OUT/(name+'.json')
            if profile.exists(): raise RuntimeError('Refusing duplicate completed profile '+name)
            cmd=command_for(SimpleNamespace(steps=10),'500M',method,'fp8_states',mb)
            cmd=[sys.executable,'-m','torch.distributed.run']+cmd[1:]
            cmd[cmd.index('src/main.py')]='scripts/memory/profile_train.py'
            cmd[cmd.index('--datasets-dir')+1]=str(DATA)
            cmd[cmd.index('--eval-cache-dir')+1]='/home/jovyan/evals_cache'
            cmd += ['--fineweb-replay-world-size','1','--fineweb-replay-layout','concat','--activation-checkpointing','none',
                    '--mem-profile',str(profile),'--mem-profile-steps','10','--mem-snapshot',str(OUT/(name+'.snapshot.pickle'))]
            (OUT/(name+'.config.json')).write_text(json.dumps(dict(command=cmd,revision=revision),indent=2))
            print('PROFILE_START='+name,flush=True)
            with (OUT/(name+'.log')).open('w') as log:
                CHILD=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while CHILD.poll() is None:
                    try:CHILD.wait(timeout=30)
                    except subprocess.TimeoutExpired:guard()
            code=CHILD.returncode;CHILD=None
            if code or not profile.exists():
                print((OUT/(name+'.log')).read_text(errors='replace')[-6000:],flush=True)
                raise RuntimeError(f'{name} failed with exit {code}')
            payload=json.loads(profile.read_text());assert len(payload['steps'])==10
            record=dict(name=name,config=payload['config'],summary=payload['summary'],revision=revision)
            results.append(record);print('PROFILE_RESULT='+json.dumps(record),flush=True)
    (OUT/'results.json').write_text(json.dumps(results,indent=2));print('PROFILE_COMPLETE=4',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;traceback.print_exc();print('PROFILE_EXIT=1',flush=True)
    else:print('PROFILE_EXIT=0',flush=True)
