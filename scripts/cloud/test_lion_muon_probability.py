"""Schedule, RNG restore, periodic parity and DDP branch agreement tests."""
import copy
import os
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[2] / 'src')]
import torch
import torch.distributed as dist
from optim.lion_muon_probability import ProbabilityLionMuon, SCHEDULES, muon_probability
from third_party.lion_muon.lion_muon import LionMuon
from third_party.lion_muon import lion_muon

n = 39250
for schedule in SCHEDULES[:-1]:
    ps = [muon_probability(schedule, i, n) for i in range(n)]
    assert all(0 <= p <= 1 for p in ps)
    assert abs(sum(ps) - 7850) < 1e-7, (schedule, sum(ps))

torch.manual_seed(0)
p = torch.nn.Parameter(torch.randn(4, 8))
a = ProbabilityLionMuon([p], total_steps=n)
global_rng = torch.get_rng_state().clone()
for i in range(100):
    a._is_muon_step()
    a._step_count += 1
assert torch.equal(global_rng, torch.get_rng_state())
b = ProbabilityLionMuon([torch.nn.Parameter(p.detach().clone())], total_steps=n)
b.load_state_dict(copy.deepcopy(a.state_dict()))
for i in range(100):
    assert a._is_muon_step() == b._is_muon_step()
    a._step_count += 1
    b._step_count += 1
print('LION_MUON_SCHEDULE_RNG_TEST_OK', flush=True)

dist.init_process_group('nccl')
rank = dist.get_rank()
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
# Unit-test exact optimizer math without compile overhead. Training uses the
# unmodified compiled upstream primitive in a separate Python process.
lion_muon.zeropower_via_newtonschulz5 = lion_muon.zeropower_via_newtonschulz5.__wrapped__
torch.manual_seed(1)
params = [torch.nn.Parameter(torch.randn(4, 8, device=device)) for _ in range(2)]
other = [torch.nn.Parameter(p.detach().clone()) for p in params]
old = LionMuon(params, lr=1e-3, weight_decay=1e-4)
new = ProbabilityLionMuon(other, total_steps=30, probability_schedule='periodic', lr=1e-3, weight_decay=1e-4)
for step in range(30):
    for p, q in zip(params, other):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
    old.step()
    new.step()
    for p, q in zip(params, other):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
assert new.muon_step_count == 6
for schedule in SCHEDULES:
    opt = ProbabilityLionMuon(other, total_steps=30, probability_schedule=schedule)
    for step in range(30):
        chosen = opt._is_muon_step()
        decision = torch.tensor(int(chosen), device=device)
        decisions = [torch.zeros_like(decision) for _ in range(dist.get_world_size())]
        dist.all_gather(decisions, decision)
        assert all(d.item() == decisions[0].item() for d in decisions)
        opt._step_count += 1
print(f'LION_MUON_PERIODIC_PARITY_DDP_TEST_OK rank={rank}', flush=True)
dist.destroy_process_group()
