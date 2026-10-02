#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PWD}/src:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
python - <<'PY'
import io
import torch

from third_party.coat.utils._fp8_quantization_config import QuantizationConfig
from optim.memory_efficient.fp8_slim_adam import FP8SlimAdamW

qargs = QuantizationConfig(
    quantize_model="none", first_order_bit="E4M3", second_order_bit="E4M3",
    first_order_expansion="expand", second_order_expansion="expand", qgroup_size=128,
)
parameter = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10)
optimizer = FP8SlimAdamW(
    [("weight", parameter)], qargs=qargs, lr=5e-4, betas=(0.9, 0.99),
    eps=1e-8, weight_decay=1e-4, layer_map_path=None, verbose=False,
)
for _ in range(3):
    parameter.grad = torch.full_like(parameter, 0.1)
    optimizer.step()
state = optimizer.state[parameter]
assert state["mu"].dtype == torch.float8_e4m3fn
assert state["nu"].dtype == torch.float8_e4m3fn
assert torch.isfinite(parameter).all()
buffer = io.BytesIO()
torch.save(optimizer.state_dict(), buffer)
buffer.seek(0)
restored = FP8SlimAdamW(
    [("weight", parameter)], qargs=qargs, lr=5e-4, betas=(0.9, 0.99),
    eps=1e-8, weight_decay=1e-4, layer_map_path=None, verbose=False,
)
restored.load_state_dict(torch.load(buffer, weights_only=False))
assert restored.state[parameter]["mu"].dtype == torch.float8_e4m3fn
assert restored.state[parameter]["nu"].dtype == torch.float8_e4m3fn
print("FP8_SLIMADAM_CPU_TEST_OK", flush=True)
PY
