"""Exercise FP32 state storage and the SoftMuon transition on CUDA."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from third_party.softsign.softmuon import SingleDeviceSoftMuonWithAuxAdam

for dtype in (torch.float32, torch.bfloat16):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 24, device='cuda', dtype=dtype))
    q = torch.nn.Parameter(torch.randn(24, device='cuda', dtype=dtype))
    opt = SingleDeviceSoftMuonWithAuxAdam([
        dict(params=[p], use_muon=True, lr=1e-3, sign_iters=1, transition_iters=3),
        dict(params=[q], use_muon=False, lr=1e-3),
    ])
    for _ in range(4):
        p.grad, q.grad = torch.randn_like(p), torch.randn_like(q)
        opt.step()
        assert torch.isfinite(p).all() and torch.isfinite(q).all()
    assert opt.param_groups[0]['schedule'] is not None
    for state in opt.state.values():
        if isinstance(state, dict):
            for key in ('momentum_buffer', 'exp_avg', 'exp_avg_sq'):
                if key in state: assert state[key].dtype == torch.float32
    print('SOFTMUON_TRANSITION_TEST_OK', dtype, flush=True)
