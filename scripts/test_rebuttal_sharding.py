"""CPU/Gloo test: state ownership preserves AdamW updates and removes replicas."""
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from stage4.rebuttal_memory import install_state_sharding


dist.init_process_group("gloo")
torch.manual_seed(123)
parameters = [torch.nn.Parameter(torch.randn(7, 5)), torch.nn.Parameter(torch.randn(3, 4)),
              torch.nn.Parameter(torch.randn(8))]
reference = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
sharded = torch.optim.AdamW(parameters, lr=0.01)
baseline = torch.optim.AdamW(reference, lr=0.01)
install_state_sharding(SimpleNamespace(optimizer=sharded))
for step in range(5):
    for p, q in zip(parameters, reference):
        gradient = torch.full_like(p, 0.1 + step)
        p.grad = gradient.clone()
        q.grad = gradient.clone()
    sharded.step()
    baseline.step()
    for p, q in zip(parameters, reference):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
count = torch.tensor(sum("exp_avg" in s for s in sharded.state.values()))
dist.all_reduce(count)
assert count.item() == len(parameters), count
if dist.get_rank() == 0:
    print("REBUTTAL_SHARDING_TEST_OK", flush=True)
dist.destroy_process_group()
