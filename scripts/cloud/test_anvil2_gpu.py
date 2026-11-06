"""CUDA/DDP ANVIL II smoke including engagement, Adam, and resume."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / 'src'))
from optim.anvil2 import Anvil2

rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
world = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
local = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
torch.cuda.set_device(local)
if world > 1:
    rendezvous = Path(os.environ['RESULTS_DIR']) / 'anvil2_test_rendezvous'
    dist.init_process_group('nccl', init_method='file://' + str(rendezvous), rank=rank, world_size=world)
torch.manual_seed(19)
params = [torch.nn.Parameter(torch.randn(shape, device='cuda'))
          for shape in ((128, 384), (384, 128), (64, 64), (128,))]
groups = [dict(params=params[:3]), dict(params=params[3:], is_proj_params=False)]
optimizer = Anvil2(groups, total_steps=6, lr=float(os.environ['LR']))
for step in range(6):
    for p in params:
        p.grad = torch.randn_like(p)
        if world > 1:
            dist.all_reduce(p.grad)
            p.grad.div_(world)
    optimizer.step()
    if step == 2:
        optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    for p in params:
        if not p.isfinite().all():
            raise RuntimeError('Nonfinite ANVIL II CUDA parameter')
        if world > 1:
            replicas = [torch.empty_like(p) for _ in range(world)]
            dist.all_gather(replicas, p)
            for other in replicas:
                torch.testing.assert_close(other, p, rtol=0, atol=0)
        for value in optimizer.state[p].values():
            if isinstance(value, torch.Tensor):
                if value.dtype != torch.float32 or not value.isfinite().all():
                    raise RuntimeError('Invalid ANVIL II CUDA state')
torch.cuda.synchronize()
print(f'ANVIL2_GPU_TEST_OK rank={rank} world={world} engage_step=3 completed_steps=6', flush=True)
if world > 1:
    dist.destroy_process_group()
