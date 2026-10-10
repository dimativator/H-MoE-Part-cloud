#!/usr/bin/env python3
"""Three independent 4-GPU precision queues, each with a verified smoke and cooldown."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import sign_ademamix_campaign as campaign


PRECISIONS = ("w8a8g8_fp32", "w8a8g8_fp8", "w16a16g16_fp8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("precision", choices=PRECISIONS)
    parser.add_argument("--output", type=Path, default=Path("/home/jovyan/dimativator/sign-ademamix-precision-20261010"))
    parser.add_argument("--dataset", type=Path, default=Path("/workspace-SR006.nfs3/dimativator/fineweb-h200-packed"))
    args = parser.parse_args()
    os.environ.update(WANDB_BASE_URL="https://wandb-radfan.ru", WANDB_ENTITY="andrey")
    campaign.CAMPAIGN = "sign-ademamix-precision-20261010"
    sys.path.insert(0, str(campaign.ROOT / "src"))
    import torch
    from types import SimpleNamespace
    from optim.optimization import get_optimizer
    qargs = SimpleNamespace(first_order_bit="E4M3", second_order_bit="E4M3",
        first_order_expansion="expand", second_order_expansion="expand", qgroup_size=128, expand_min=16)
    parameter = torch.nn.Parameter(torch.ones(256, device="cuda", dtype=torch.float32))
    config = SimpleNamespace(opt="ademamix_sign", fp8_optim=args.precision in ("w8a8g8_fp8", "w16a16g16_fp8"),
        lr=1e-4, beta1=0.9, ademamix_sign_beta3=0.9999, ademamix_sign_alpha=8,
        ademamix_sign_beta3_warmup_steps=150914, ademamix_sign_alpha_warmup_steps=150914, weight_decay=1e-4)
    optimizer = get_optimizer([{"params": [parameter]}], config, qargs=qargs)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    expected = torch.float8_e4m3fn if config.fp8_optim else torch.float32
    for key in ("exp_avg_fast", "exp_avg_slow"):
        assert optimizer.state[parameter][key].dtype == expected
    campaign.emit("SIGN_STATE_DTYPES", {"precision": args.precision, "momentum_dtype": str(expected)})
    del optimizer, parameter
    torch.cuda.empty_cache()
    metadata = json.loads((args.dataset / "packed_metadata.json").read_text())
    assert int(metadata["iterations"]) >= 150914
    assert int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) in (1, 4)
    if shutil.disk_usage(args.output.parent).free < 9_000_000_000:
        raise RuntimeError("Need 9 GB checkpoint headroom")
    common = dict(output=args.output, dataset=args.dataset, world=4, precision=args.precision, run_date="20261010")
    campaign.run_one("500m", "1e-4", 2, args.precision + "_smoke4gpu", smoke=True, **common)
    trunk = campaign.run_one("500m", "1e-4", 150914, args.precision + "_full2xc", milestones=(67911,), **common)
    campaign.run_one("500m", "1e-4", 75457, args.precision + "_decay1xc", resume=trunk / "ckpts/67911", **common)
    if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
        campaign.cleanup_checkpoint(args.output, trunk)
        campaign.emit("SIGN_PRECISION_COMPLETE", {"precision": args.precision,
            "checkpoints_remaining": [str(p) for p in trunk.rglob("*.pt")]})


if __name__ == "__main__":
    main()
