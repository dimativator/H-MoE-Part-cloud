from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from optim.memory_efficient.fp8_projected_adam import FP8APOLLOAdamW, FP8GaloreAdamW


@pytest.mark.parametrize("optimizer_cls", [FP8APOLLOAdamW, FP8GaloreAdamW])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("projected", [False, True])
def test_fp8_moments_survive_checkpoint_roundtrip(optimizer_cls, dtype, projected):
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(4, 4, dtype=dtype))
    qargs = SimpleNamespace(
        qgroup_size=8, first_order_bit="E4M3", second_order_bit="E4M3",
        first_order_expansion="expand", second_order_expansion="expand", expand_min=16,
    )
    group = {"params": [param], "is_proj_params": projected}
    if projected and optimizer_cls is FP8APOLLOAdamW:
        group.update(rank=1, update_proj_gap=50, scale=1, proj="random",
                     scale_type="channel", proj_type="std")
    groups = [group]
    optimizer = optimizer_cls(groups, qargs=qargs, lr=1e-3)
    for _ in range(2):
        param.grad = torch.randn_like(param)
        optimizer.step()
    saved = deepcopy(optimizer.state_dict())
    restored = optimizer_cls(groups, qargs=qargs, lr=1e-3)
    restored.load_state_dict(saved)
    state = restored.state[param]
    for name in ("exp_avg", "exp_avg_sq"):
        assert state[name].dtype == torch.float8_e4m3fn
        assert torch.isfinite(state[name].float()).all()
        assert torch.isfinite(state[f"scale_{name}"].float()).all()
    assert state["step"] == 2
    param.grad = torch.randn_like(param)
    restored.step()
    assert restored.state[param]["step"] == 3
    assert torch.isfinite(param).all()
