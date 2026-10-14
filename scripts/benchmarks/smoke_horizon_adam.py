"""Train the repository's small Llama on synthetic tokens without downloads.

Run from any directory:
    python scripts/benchmarks/smoke_horizon_adam.py
    python scripts/benchmarks/smoke_horizon_adam.py --device cuda:0 --dtype bfloat16

Accepts the normal training CLI options. Defaults are deliberately tiny. This
checks integration and finite updates, not generalization or LLM-scale quality.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config.base import parse_args
from models.llama import Llama
from optim.horizon_adam import build_horizon_adam


def main() -> None:
    defaults = [
        "--experiment-name", "horizon_adam_smoke", "--opt", "horizon_adam",
        "--device", "cpu", "--dtype", "float32", "--iterations", "12",
        "--model", "llama", "--n-layer", "2", "--n-embd", "64", "--n-head", "4",
        "--multiple-of", "32", "--vocab-size", "128", "--sequence-length", "32",
        "--batch-size", "2", "--lr", "0.003", "--weight-decay", "0.01",
        "--horizon-adam-horizon", "2", "--horizon-adam-block-size", "32",
        "--horizon-adam-probes", "16", "--horizon-adam-min-size", "128",
    ]
    args = parse_args(ArgumentParser(description=__doc__), defaults + sys.argv[1:], None)
    if args.fp8 or args.opt != "horizon_adam":
        raise ValueError("This smoke script supports HorizonAdam with ordinary activations.")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(1)
    # Match src/main.py: FP32 master parameters with mixed-precision forward.
    model = Llama(args).to(device=args.device)
    opt = build_horizon_adam(model.parameters(), args)
    tokens = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length + 1), device=args.device)
    inputs, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    losses = []
    for _ in range(args.iterations):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(torch.device(args.device).type, dtype=getattr(torch, args.dtype),
                            enabled=args.dtype != "float32"):
            loss = model(inputs, targets)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite HorizonAdam smoke loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(loss.item())
    parameter_count = sum(p.numel() for p in model.parameters())
    print(json.dumps(dict(
        device=args.device, dtype=args.dtype, steps=args.iterations,
        first_loss=losses[0], last_loss=losses[-1], parameters=parameter_count,
        state_bytes=opt.state_size_bytes(), fp32_adam_state_bytes=8 * parameter_count,
        codec_updates=sum(state.get("codec_updates", 0) for state in opt.state.values()),
    ), indent=2))


if __name__ == "__main__":
    main()
