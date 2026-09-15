from types import SimpleNamespace

import torch

from optim.memory_efficient.fp8_slim_adam import FP8SlimAdamW


def test_fp8_slim_adam_persists_quantized_moments_across_steps():
    qargs = SimpleNamespace(
        qgroup_size=8,
        first_order_bit="E4M3",
        second_order_bit="E4M3",
        first_order_expansion="expand",
        second_order_expansion="expand",
        expand_min=16,
    )
    param = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = FP8SlimAdamW(
        [("weight", param)],
        qargs=qargs,
        lr=5e-4,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.1,
        rules_json_path=None,
        layer_map_path=None,
        verbose=False,
    )

    for _ in range(2):
        param.grad = torch.randn_like(param)
        optimizer.step()
        state = optimizer.state[param]
        assert state["mu"].dtype == torch.float8_e4m3fn
        assert state["nu"].dtype == torch.float8_e4m3fn
        assert state["scale_mu"].dtype == torch.float32
        assert state["scale_nu"].dtype == torch.float32
