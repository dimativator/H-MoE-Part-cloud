#!/usr/bin/env python3
"""257M clipping=1 ablation, with offline W&B and exact local final checks."""
import json
import math
import os
from pathlib import Path
import subprocess
import sign_ademamix_campaign as campaign

root = Path('/home/jovyan/dimativator/sign-ademamix-clip1-20261011')
group = 'sign-ademamix-clip1-20261011-257m'
name = '257m_ademamix_sign_lr1e-4_clip1_1xc'
run_id = 'sas-clip1-1xc-257m-1e-4-20261011'
experiment = root / group / name
if (experiment / 'local_verified.json').exists():
    print((experiment / 'local_verified.json').read_text(), flush=True)
    raise SystemExit(0)
if (experiment / 'metrics.jsonl').exists():
    raise RuntimeError('Existing metrics: inspect before retrying')
experiment.mkdir(parents=True, exist_ok=True)
(root / 'wandb').mkdir(exist_ok=True)
os.environ.update(WANDB_MODE='offline', WANDB_BASE_URL='https://wandb-radfan.ru',
    WANDB_ENTITY='andrey', WANDB_RUN_ID=run_id, WANDB_DIR=str(root / 'wandb'),
    RANK='0', WORLD_SIZE='1', LOCAL_RANK='0', MASTER_ADDR='127.0.0.1',
    MASTER_PORT='29500', PYTHONUNBUFFERED='1',
    TOKENIZERS_PARALLELISM='false', TRITON_CACHE_DIR='/tmp/sign-clip1-20261011')
command = campaign.command('257m', '1e-4', 39250, name, group, root,
    Path('/workspace-SR006.nfs3/dimativator/fineweb-h200-packed'), 1)
command[command.index('--grad-clip') + 1] = '1.0'
(experiment / 'launch.json').write_text(json.dumps({'command':command,'run_id':run_id,'wandb_mode':'offline'}, indent=2))
print('SIGN_START '+json.dumps({'run_id':run_id,'grad_clip':1.0,'target':39250,'gpus':1}), flush=True)
with (experiment / 'rank0.log').open('w') as log:
    subprocess.run(command, cwd=campaign.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
rows = [json.loads(line) for line in (experiment / 'metrics.jsonl').read_text().splitlines() if line.strip()]
finals = [row for row in rows if row.get('iter') == 39250]
assert {'validation','downstream','lm_eval'}.issubset({r.get('event') for r in finals})
losses = [r['final-val/loss'] for r in finals if 'final-val/loss' in r]
assert len(losses)==1 and math.isfinite(losses[0])
assert all(r.get('consumed_tokens') == 39250*128*1024 for r in finals if r.get('event')=='validation')
result={'run_id':run_id,'target':39250,'grad_clip':1.0,'final_val_loss':losses[0],'status':'local_verified','wandb_sync':'pending'}
(experiment / 'local_verified.json').write_text(json.dumps(result,indent=2))
print('SIGN_RESULT '+json.dumps(result),flush=True)
