"""Peak-memory sweep: optimizer × state precision × compute precision × microbatch ×
activation checkpointing × activation offload.

Each case runs src/main.py for a few steps with --mem-profile and writes
<out-dir>/<case>.json (see src/optim/mem_profile.py). Existing results are skipped,
so an interrupted sweep resumes where it stopped. Cases that run out of memory are
recorded with status "oom".

Arguments after "--" are passed to every run (dataset paths and the like). Example,
one GPU, 500M model, Table 1/3 grid under three checkpointing modes:

    python scripts/memory/sweep.py --out-dir results/mem500m \
        --ac none,mlp,full --offload off,on -- \
        --dataset fineweb --datasets-dir /path/to/fineweb-packed \
        --fineweb-replay-world-size 1 --fineweb-replay-layout concat \
        --eval-cache-dir /path/to/evals_cache --streaming --workers 8

Then: python scripts/memory/collect.py results/mem500m
"""

import argparse
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Hyperparameters do not change memory, but these match the 500M launch scripts
# (run_optimizer_fp8_cloud.sh, run_500m_bf16_wd_tuning_2gpu.sh,
# run_slimadam_500m_precisions_4gpu.sh) so the optimizer code paths are the same.
PROJ_ARGS = ["--density", "0.25", "--update_gap", "50", "--coord_choice", "columns",
             "--inactive_update_rule", "sign_sgd"]
OPTIMIZERS = {
    "adamw": ["--opt", "adamw", "--lr", "1e-3", "--beta2", "0.99"],
    "muon": ["--opt", "muon", "--lr", "1e-3", "--beta2", "0.99"],
    "soap": ["--opt", "soap", "--lr", "1e-3", "--beta2", "0.99"],
    "ademamix": ["--opt", "ademamix", "--lr", "1e-3", "--beta2", "0.999",
                 "--ademamix_beta3", "0.9999", "--ademamix_alpha", "8",
                 "--ademamix_beta3_warmup_steps", "1000",
                 "--ademamix_alpha_warmup_steps", "1000", "--grad-clip", "0.5"],
    "slimadam": ["--opt", "slim_adam", "--lr", "5e-4", "--beta2", "0.99"],
    "galore": ["--opt", "galore_adamw", "--lr", "1e-3", "--beta2", "0.999",
               "--proj_type", "svd", "--proj_side", "std", *PROJ_ARGS],
    "frugal": ["--opt", "coord_adamw", "--lr", "1e-3", "--beta2", "0.999", *PROJ_ARGS],
    "frugal_mm": ["--opt", "coord_muon", "--lr", "2e-3", "--beta2", "0.99",
                  "--non_proj_opt", "adamw", "--momentum", "0.95", "--nesterov",
                  "--muon_ns_steps", "5", *PROJ_ARGS],
    "apollo": ["--opt", "apollo_adamw", "--lr", "2e-3", "--beta2", "0.999",
               "--apollo_proj", "random", "--apollo_scale_type", "channel",
               "--apollo_scale", "1", "--proj_side", "std", *PROJ_ARGS],
}
# FP8 AdamW states use the fused Triton COAT kernel, as in the time-matched runs.
FP8_STATE_OPT_OVERRIDE = {"adamw": ["--opt", "triton_coat_adamw"]}
# FP8 optimizer states are wired on this branch only for these optimizers.
FP8_STATES_SUPPORTED = {"adamw", "muon", "soap", "ademamix", "slimadam", "frugal_mm"}

FP8_STATE_ARGS = ["--fp8-optim", "--fp8-qgroup-size", "128", "--fp8-first-order-bit", "E4M3",
                  "--fp8-second-order-bit", "E4M3", "--fp8-expansion", "expand"]
COMPUTE_ARGS = {
    "bf16": [],  # W16A16G32
    "fp8": ["--fp8", "--fp8-fabit", "E4M3", "--fp8-fwbit", "E4M3", "--fp8-babit", "E5M2",
            "--fp8-bwbit", "E5M2", "--fp8-group-size", "16"],  # W8A8G8
}
MODEL_500M = "--n-layer 18 --n-embd 1280 --n-head 20 --multiple-of 256"


def csv_list(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--optimizers", type=csv_list, default=list(OPTIMIZERS))
    p.add_argument("--states", type=csv_list, default=["fp32", "fp8"], help="fp32,fp8")
    p.add_argument("--compute", type=csv_list, default=["bf16"], help="bf16,fp8")
    p.add_argument("--microbatches", type=lambda s: [int(v) for v in csv_list(s)],
                   default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--ac", type=csv_list, default=["none", "mlp", "full"], help="none,mlp,full")
    p.add_argument("--offload", type=csv_list, default=["off"], help="off,on")
    p.add_argument("--effective-batch", type=int, default=128,
                   help="Sequences per optimizer step per GPU; acc-steps = this / microbatch.")
    p.add_argument("--acc-steps", type=int, default=None,
                   help="Fixed accumulation steps (overrides --effective-batch).")
    p.add_argument("--steps", type=int, default=3, help="Profiled optimizer steps per case.")
    p.add_argument("--snapshot", action="store_true",
                   help="Also dump a torch.cuda.memory snapshot of the last step per case.")
    p.add_argument("--model-args", default=MODEL_500M)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--launcher", default=f"{shlex.quote(sys.executable)} src/main.py",
                   help='e.g. "torchrun --standalone --nproc_per_node=1 src/main.py"')
    p.add_argument("--timeout", type=int, default=3600, help="Seconds per case.")
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help='Arguments after "--" are passed to every run.')
    args = p.parse_args()
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    unknown = set(args.optimizers) - set(OPTIMIZERS)
    if unknown:
        p.error(f"unknown optimizers: {sorted(unknown)}")
    return args


def case_name(opt, states, compute, mb, ac, offload):
    return f"{opt}__{states}-states__{compute}__mb{mb}__ac-{ac}__offload-{offload}"


def build_command(args, opt, states, compute, mb, ac, offload, json_path):
    if args.acc_steps is not None:
        acc = args.acc_steps
    else:
        if args.effective_batch % mb:
            raise ValueError(f"effective batch {args.effective_batch} not divisible by {mb}")
        acc = args.effective_batch // mb
    name = case_name(opt, states, compute, mb, ac, offload)
    cmd = shlex.split(args.launcher) + [
        "--experiment-name", f"mem_{name}",
        "--results-base-folder", str(args.out_dir / "exps"), "--no-local-save",
        "--model", "llama", *shlex.split(args.model_args),
        "--sequence-length", str(args.sequence_length), "--dtype", "bfloat16",
        "--seed", "0", "--weight-decay", "1e-4", "--beta1", "0.9", "--grad-clip", "1.0",
        "--scheduler", "none", "--warmup-steps", "0", "--iterations", "100000",
        "--eval-interval", "1000000", "--eval-batches", "1", "--log-interval", "1000000",
        "--batch-size", str(mb), "--acc-steps", str(acc),
        "--activation-checkpointing", ac,
        "--mem-profile", str(json_path), "--mem-profile-steps", str(args.steps),
        *COMPUTE_ARGS[compute],
        *OPTIMIZERS[opt],
    ]
    if states == "fp8":
        cmd += FP8_STATE_OPT_OVERRIDE.get(opt, []) + FP8_STATE_ARGS
    if offload == "on":
        cmd.append("--activation-offload")
    if args.snapshot:
        cmd += ["--mem-snapshot", str(json_path.with_suffix(".snapshot.pickle"))]
    return cmd + args.extra


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    grid = itertools.product(args.optimizers, args.states, args.compute,
                             args.microbatches, args.ac, args.offload)
    counts = {"done": 0, "skipped": 0, "oom": 0, "failed": 0, "unsupported": 0}
    for opt, states, compute, mb, ac, offload in grid:
        name = case_name(opt, states, compute, mb, ac, offload)
        json_path = args.out_dir / f"{name}.json"
        if states == "fp8" and opt not in FP8_STATES_SUPPORTED:
            print(f"SKIP {name}: FP8 states are not wired for {opt} on this branch")
            counts["unsupported"] += 1
            continue
        if json_path.exists():
            counts["skipped"] += 1
            continue
        cmd = build_command(args, opt, states, compute, mb, ac, offload, json_path)
        if args.dry_run:
            print(shlex.join(cmd))
            continue
        print(f"RUN  {name}", flush=True)
        log_path = args.out_dir / "logs" / f"{name}.log"
        try:
            proc = subprocess.run(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, timeout=args.timeout)
            output, returncode = proc.stdout, proc.returncode
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
            returncode = "timeout"
        log_path.write_text(output)
        if returncode == 0 and json_path.exists():
            counts["done"] += 1
            continue
        oom = "out of memory" in output.lower()
        status = "oom" if oom else "failed"
        counts[status] += 1
        stub = {
            "status": status,
            "returncode": returncode,
            "config": {"opt_key": opt, "states": states, "compute": compute,
                       "batch_size": mb, "activation_checkpointing": ac,
                       "activation_offload": offload == "on"},
            "log": str(log_path),
        }
        # OOM is a result; other failures are not written, so a rerun retries them.
        if oom:
            json_path.write_text(json.dumps(stub, indent=2))
        print(f"{status.upper()} {name} (see {log_path})", flush=True)
    print(" ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
