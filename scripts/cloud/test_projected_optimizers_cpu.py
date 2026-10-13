"""Check production factory and checkpoint roundtrips without allocating GPUs."""
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from optim.optimization import get_optimizer
from optim.memory_efficient.fp8_projected_adam import FP8APOLLOAdamW, FP8GaloreAdamW


def main() -> None:
    print("TORCH_VERSION", torch.__version__, flush=True)
    qargs = SimpleNamespace(qgroup_size=128, first_order_bit="E4M3",
                            second_order_bit="E4M3", first_order_expansion="expand",
                            second_order_expansion="expand", expand_min=16)
    for name, fp8_cls in (("apollo_adamw", FP8APOLLOAdamW), ("galore_adamw", FP8GaloreAdamW)):
        for fp8 in (False, True):
            for dtype in (torch.float32, torch.bfloat16):
                torch.manual_seed(0)
                param = torch.nn.Parameter(torch.randn(8, 8, dtype=dtype))
                args = SimpleNamespace(opt=name, fp8_optim=fp8, non_proj_opt="adamw",
                                       beta1=0.9, beta2=0.999, lr=1e-3, weight_decay=1e-2,
                                       eps=1e-8, n_embd=8, density=0.25, rank=2,
                                       update_gap=50, proj_params_lr_scale=1.0,
                                       reset_statistics=True, inactive_update_rule="sign_sgd",
                                       inactive_lr_scale=1.0, proj_side="std", proj_type="svd",
                                       apollo_scale=1.0, apollo_proj="random",
                                       apollo_scale_type="channel", apollo_scale_front=False)
                groups = [{"params": [param], "is_proj_params": True}]
                optimizer = get_optimizer(groups, args, qargs=qargs if fp8 else None)
                assert isinstance(optimizer, fp8_cls) == fp8
                for _ in range(2):
                    param.grad = torch.randn_like(param)
                    optimizer.step()
                restored = get_optimizer(groups, args, qargs=qargs if fp8 else None)
                restored.load_state_dict(deepcopy(optimizer.state_dict()))
                for key in ("exp_avg", "exp_avg_sq"):
                    expected = torch.float8_e4m3fn if fp8 else dtype
                    assert restored.state[param][key].dtype == expected
                param.grad = torch.randn_like(param)
                restored.step()
                assert torch.isfinite(param).all()
                print("FACTORY_ROUNDTRIP_PASS", name, fp8, str(dtype), flush=True)
    print("PROJECTED_FACTORY_CPU_TESTS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
