"""Two-rank CUDA gate for both branches and the exponent transition."""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from optim.power_lion_muon import PowerLionMuon


def main():
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    try:
        for probability in (0., 1., 0.2):
            p = torch.nn.Parameter(torch.eye(32, device='cuda'))
            opt = PowerLionMuon([{'params': [p], 'use_power': True}], total_steps=5,
                                start_fraction=0., probability=probability,
                                probability_seed=20261020, weight_decay=0.1)
            for step in range(5):
                p.grad = torch.eye(32, device='cuda') * (step + 1.)
                opt.step()
                assert torch.isfinite(p).all()
                assert opt.state[p]['momentum_buffer'].dtype == torch.float32
                copies = [torch.empty_like(p) for _ in range(dist.get_world_size())]
                dist.all_gather(copies, p)
                torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
            assert opt.last_exponent == 1.
        print(f'POWER_LION_MUON_GPU_TEST_OK rank={rank} branches=both exponent=0_to_1 states=FP32', flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
