"""Exercise hard/soft branches and assert synchronized DDP optimizer state."""

import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from optim.soft_lion_muon import SoftLionMuon, elementwise_softsign, spectral_softsign

rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
world = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
local = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
torch.cuda.set_device(local)
if world > 1:
    rendezvous = Path(os.environ['RESULTS_DIR']) / 'optimizer_test_rendezvous'
    dist.init_process_group('nccl', init_method='file://' + str(rendezvous), rank=rank, world_size=world)
torch.manual_seed(19)
p = torch.nn.Parameter(torch.randn(32, 48, device='cuda'))
optimizer = SoftLionMuon([p], total_steps=10, start_fraction=0.3, period=2)
for step in range(10):
    p.grad = torch.randn_like(p)
    if world > 1:
        dist.all_reduce(p.grad)
        p.grad.div_(world)
    optimizer.step()
    if not p.isfinite().all():
        raise RuntimeError('Nonfinite GPU parameter')
    if world > 1:
        replicas = [torch.empty_like(p) for _ in range(world)]
        dist.all_gather(replicas, p)
        for other in replicas:
            torch.testing.assert_close(other, p, rtol=0, atol=0)
for shape in ((128, 384), (384, 128)):
    matrix = torch.randn(shape, device='cuda') * 0.01
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    for tau in (1.0, 100.0, 1e6):
        expected = (u * elementwise_softsign(s, tau)) @ vh
        actual = spectral_softsign(matrix, tau)
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-3)
torch.cuda.synchronize()
start = time.monotonic()
spectral_softsign(torch.randn(1024, 2816, device='cuda') * 1e-3, 1e4)
torch.cuda.synchronize()
print(f'SOFT_LION_MUON_GPU_TEST_OK rank={rank} world={world} spectral_seconds={time.monotonic()-start:.4f}', flush=True)
if world > 1:
    dist.destroy_process_group()
