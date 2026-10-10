#!/usr/bin/env python3
"""Cloud.ru Sign-AdEMAMix LR queues and sequential cooldown experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "sign-ademamix-20261007"
LR_QUEUES = {"A": ["2e-3", "5e-4"], "B": ["1e-3", "1e-4"]}
BUDGETS = {"257m": {1: 39250, 2: 78500, 4: 157000}, "500m": {1: 75457, 2: 150914}}


def emit(kind: str, data: dict) -> None:
    print(kind + " " + json.dumps(data, sort_keys=True), flush=True)


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def command(size: str, lr: str, target: int, name: str, group: str,
            output: Path, dataset: Path, world: int, *, resume: Path | None = None,
            milestones: tuple[int, ...] = (), smoke: bool = False,
            precision: str | None = None) -> list[str]:
    layers, width, heads = (12, 1024, 8) if size == "257m" else (18, 1280, 20)
    wd = "0.1" if size == "257m" else "1e-4"
    horizon = BUDGETS[size][4 if size == "257m" and (target > 39250 or resume) else 2 if size == "500m" else 1]
    result = [sys.executable, "src/main.py", "--distributed-backend", "nccl",
        "--seed", "0", "--data-seed", "1337", "--dataset", "fineweb",
        "--datasets-dir", str(dataset), "--fineweb-replay-world-size", "2" if world == 1 else "1",
        "--fineweb-replay-layout", "concat", "--streaming", "--workers", "8",
        "--eval-cache-dir", os.environ.get("EVAL_CACHE_DIR", "/home/jovyan/evals_cache"),
        "--model", "llama", "--n-layer", str(layers), "--n-embd", str(width),
        "--n-head", str(heads), "--multiple-of", "256", "--sequence-length", "1024",
        "--dtype", "bfloat16", "--opt", "ademamix_sign", "--lr", lr,
        "--weight-decay", wd, "--beta1", "0.9", "--grad-clip", "0.5",
        "--ademamix_sign_beta3", "0.9999", "--ademamix_sign_alpha", "8",
        "--ademamix_sign_beta3_warmup_steps", str(horizon),
        "--ademamix_sign_alpha_warmup_steps", str(horizon),
        "--batch-size", str(32 // world), "--acc-steps", str(4 * world),
        "--eval-batch-size", "32", "--scheduler", "none" if smoke else "wsd",
        "--warmup-steps", "0" if smoke or resume else "2000", "--iterations", str(target),
        "--wsd-fract-decay", "1.0" if resume else "0.1", "--wsd-final-lr-scale", "0",
        "--decay-type", "cosine", "--eval-interval", "2" if smoke else "500",
        "--eval-batches", "1" if smoke else "32", "--log-interval", "1" if smoke else "50",
        "--latest-ckpt-interval", "0", "--permanent-ckpt-interval", "0",
        "--experiment-name", name, "--results-base-folder", str(output),
        "--metrics-jsonl", str(output / group / name / "metrics.jsonl"),
        "--wandb", "--wandb-project", "fp8-pretrain", "--wandb-group", group,
        "--wandb-tags", "Sign-AdEMAMix", CAMPAIGN, size, "BF16", f"lr{lr}", f"{world}gpu"]
    if precision in ("w8a8g8_fp32", "w8a8g8_fp8"):
        result += ["--fp8", "--fp8-fabit", "E4M3", "--fp8-fwbit", "E4M3",
                   "--fp8-babit", "E5M2", "--fp8-bwbit", "E5M2", "--fp8-group-size", "16"]
    if precision in ("w8a8g8_fp8", "w16a16g32_fp8"):
        result += ["--fp8-optim", "--fp8-qgroup-size", "128", "--fp8-first-order-bit", "E4M3",
                   "--fp8-second-order-bit", "E4M3", "--fp8-expansion", "expand"]
    if precision not in (None, "w8a8g8_fp32", "w8a8g8_fp8", "w16a16g16_fp32", "w16a16g32_fp8"):
        raise ValueError(f"Unknown precision: {precision}")
    if not smoke:
        result += ["--downstream-eval-enabled", "--downstream-eval-interval", "2000",
            "--downstream-task-group", "basic_v2", "--lm-eval-enabled", "--lm-eval-interval", "2000",
            "--lm-eval-datasets", "wikitext103"]
    if milestones:
        result += ["--inter-ckpts", *(str(value) for value in milestones)]
    else:
        result += ["--no-local-save"]
    if resume:
        result += ["--resume-from", str(resume), "--decay-from-checkpoint"]
    return result


def verify(metrics: Path, target: int, run_id: str, *, smoke: bool = False, precision: str | None = None) -> dict:
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    finals = [row for row in rows if row.get("iter") == target]
    required = {"validation"} if smoke else {"validation", "downstream", "lm_eval"}
    if not required.issubset({row.get("event") for row in finals}):
        raise RuntimeError(f"Missing exact final evaluation events: {metrics}, target={target}")
    losses = [float(row["final-val/loss"]) for row in finals if "final-val/loss" in row]
    if len(losses) != 1 or not math.isfinite(losses[0]):
        raise RuntimeError("Missing finite final validation loss")
    if any(row.get("consumed_tokens") != target * 128 * 1024
           for row in finals if row.get("event") == "validation"):
        raise RuntimeError("Final token budget does not match the requested horizon")
    import wandb
    api = wandb.Api(timeout=30)
    for attempt in range(6):
        api.flush()
        run = api.run(f"andrey/fp8-pretrain/{run_id}")
        remote = run.summary.get("final-val/loss")
        if isinstance(remote, (int, float)) and math.isclose(remote, losses[0], rel_tol=0, abs_tol=1e-12):
            break
        if attempt == 5:
            raise RuntimeError(f"W&B final not verified: {run_id}")
        time.sleep(5)
    assert run.config["opt"] == "ademamix_sign"
    assert run.config["iterations"] == target
    assert run.config["dtype"] == "bfloat16"
    assert run.config["fp8"] == (precision in ("w8a8g8_fp32", "w8a8g8_fp8"))
    assert run.config["fp8_optim"] == (precision in ("w8a8g8_fp8", "w16a16g32_fp8"))
    assert run.config["batch_size"] * run.config["acc_steps"] * run.config["world_size"] == 128
    return {"run_id": run_id, "target": target, "final_val_loss": losses[0],
            "url": f"https://wandb-radfan.ru/andrey/fp8-pretrain/runs/{run_id}", "status": "verified"}


def cleanup_checkpoint(output: Path, experiment: Path) -> None:
    checkpoints = experiment / "ckpts"
    if not checkpoints.exists():
        return
    if checkpoints.is_symlink() or not checkpoints.resolve().is_relative_to(output.resolve()):
        raise RuntimeError(f"Unsafe checkpoint cleanup path: {checkpoints}")
    shutil.rmtree(checkpoints)
    emit("SIGN_CHECKPOINTS_REMOVED", {"path": str(checkpoints)})


def nonfinite_loss(metrics: Path, log: Path | None = None) -> dict | None:
    if not metrics.exists():
        return None
    for line in metrics.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ("train/loss", "val/loss", "final-val/loss"):
            value = row.get(key)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                return {"step": row.get("iter"), "metric": key, "value": str(value)}
    # The strict JSON logger can fail before writing the offending Inf/NaN.
    # Classify only this explicit numerical exception, never arbitrary failures.
    if log is not None and log.exists():
        text = log.read_text(errors="replace")
        if "_log_local_metric" in text and any(
                f"ValueError: Out of range float values are not JSON compliant: {value}" in text
                for value in ("inf", "-inf", "nan")):
            return {"metric": "local_metric", "value": "nonfinite",
                    "reason": "Strict JSON metric serialization rejected Inf/NaN"}
    return None


def training_launcher(launch: list[str], world: int, mpi_size: int) -> list[str]:
    if mpi_size == 1 and world > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc_per_node={world}", *launch[1:]]
    return launch


def run_one(size: str, lr: str, target: int, suffix: str, output: Path, dataset: Path,
            world: int, *, resume: Path | None = None, milestones: tuple[int, ...] = (),
            smoke: bool = False, precision: str | None = None, run_date: str = "20261007") -> Path:
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    group = CAMPAIGN + ("-smoke" if smoke else "-" + size)
    name = f"{size}_ademamix_sign_lr{lr}_{suffix}"
    run_id = f"sas-{suffix}-{size}-{lr}-{run_date}" + ("-smoke" if smoke else "")
    experiment = output / group / name
    done = experiment / "verified.json"
    diverged = experiment / "diverged.json"
    failed = experiment / "failed.json"
    if diverged.exists():
        if rank == 0:
            emit("SIGN_RESULT", json.loads(diverged.read_text()))
        return experiment
    if done.exists():
        if rank == 0:
            emit("SIGN_RESULT", json.loads(done.read_text()))
        return experiment
    if rank == 0 and world == 1 and not smoke and (experiment / "metrics.jsonl").exists():
        numerical = nonfinite_loss(experiment / "metrics.jsonl", experiment / "rank0.log")
        if numerical:
            result = {"status": "diverged", "run_id": run_id, "lr": lr, "size": size,
                      "name": name, "diagnostic": numerical,
                      "url": f"https://wandb-radfan.ru/andrey/fp8-pretrain/runs/{run_id}"}
            atomic_json(diverged, result)
            emit("SIGN_RESULT", result)
            return experiment
    if rank == 0 and (experiment / "metrics.jsonl").exists():
        raise RuntimeError(f"Refusing to overwrite existing metrics: {experiment}")
    experiment.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(WANDB_BASE_URL="https://wandb-radfan.ru", WANDB_ENTITY="andrey", WANDB_MODE="online",
        WANDB_RUN_ID=run_id, WANDB_RESUME="never", WANDB_DIR=str(output / "wandb"),
        PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", FINEWEB_LOG_DATA_HASHES="1",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        TRITON_CACHE_DIR=f"/tmp/triton-sign-{run_date}-{suffix}-rank{rank}",
        RANK=str(rank), WORLD_SIZE=str(world),
        LOCAL_RANK=os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"),
        MASTER_ADDR=os.environ.get("MASTER_ADDR", socket.gethostname()),
        MASTER_PORT=os.environ.get("MASTER_PORT", "29500"))
    (output / "wandb").mkdir(exist_ok=True)
    launch = command(size, lr, target, name, group, output, dataset, world,
                     resume=resume, milestones=milestones, smoke=smoke, precision=precision)
    # mlsub can allocate two GPUs to one MPI worker. Spawn both training ranks
    # locally in that case, as in the existing Huawei launchers.
    launch = training_launcher(launch, world, int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    if rank == 0:
        emit("SIGN_START", {"name": name, "run_id": run_id, "target": target, "gpus": world})
        atomic_json(experiment / "launch.json", {"command": launch, "run_id": run_id, "gpus": world})
    log = experiment / f"rank{rank}.log"
    numerical = None
    with log.open("w") as stream:
        process = subprocess.Popen(launch, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            # A NaN/Inf run cannot contribute to LR selection. Stop only that
            # single-GPU tuning process, retain diagnostics, then try the next LR.
            if world == 1 and not smoke:
                numerical = nonfinite_loss(experiment / "metrics.jsonl")
                if numerical:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
            time.sleep(5)
        status = process.wait()
    if world == 1 and not smoke:
        numerical = numerical or nonfinite_loss(experiment / "metrics.jsonl", log)
    if numerical:
        result = {"status": "diverged", "run_id": run_id, "lr": lr, "size": size,
                  "name": name, "diagnostic": numerical,
                  "url": f"https://wandb-radfan.ru/andrey/fp8-pretrain/runs/{run_id}"}
        atomic_json(diverged, result)
        emit("SIGN_RESULT", result)
        return experiment
    if status:
        if rank == 0:
            atomic_json(failed, {"status": "failed", "exit_code": status, "run_id": run_id})
        print(log.read_text(errors="replace")[-16000:], flush=True)
        raise RuntimeError(f"Training process failed: {name}, rank={rank}, code={status}")
    if rank == 0:
        try:
            result = verify(experiment / "metrics.jsonl", target, run_id, smoke=smoke, precision=precision)
        except Exception:
            atomic_json(failed, {"status": "verification_failed", "run_id": run_id})
            raise
        result.update(name=name, lr=lr, size=size, metrics=str(experiment / "metrics.jsonl"))
        atomic_json(done, result)
        emit("SIGN_RESULT", result)
    else:
        for _ in range(150):
            if done.exists():
                break
            if failed.exists():
                raise RuntimeError("Rank-0 verification failed")
            time.sleep(2)
        if not done.exists():
            raise RuntimeError("Timed out waiting for rank-0 final verification")
    return experiment


def probe(dataset: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from optim.optimization import get_optimizer
    from types import SimpleNamespace
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    args = SimpleNamespace(opt="ademamix_sign", non_proj_opt="adamw", fp8_optim=False,
        lr=0.001, beta1=0.9, ademamix_sign_beta3=0.9999, ademamix_sign_alpha=8,
        ademamix_sign_beta3_warmup_steps=39250, ademamix_sign_alpha_warmup_steps=39250, weight_decay=0.1)
    optimizer = get_optimizer([{"params": [parameter]}], args)
    parameter.grad = torch.tensor([2.0])
    optimizer.step()
    metadata = json.loads((dataset / "packed_metadata.json").read_text())
    segments = []
    for rank in metadata["ranks"]:
        for segment in rank.get("segments", [rank]):
            path = dataset / segment["file"]
            assert path.is_file() and path.stat().st_size == segment["bytes"]
            segments.append({"path": str(path), "bytes": segment["bytes"]})
    import wandb
    viewer = wandb.Api(timeout=30).viewer
    emit("SIGN_PROBE", {"torch": torch.__version__, "optimizer": type(optimizer).__name__,
        "dataset": str(dataset), "iterations": metadata["iterations"], "segments": segments,
        "metadata": metadata, "wandb_authenticated": bool(viewer),
        "disk": {str(path): shutil.disk_usage(path).free for path in
                 [Path("/home/jovyan"), Path("/workspace-SR006.nfs2"), Path("/workspace-SR006.nfs3")]}})


def audit(output: Path) -> None:
    for launch_path in sorted(output.rglob("launch.json")):
        experiment = launch_path.parent
        launch = json.loads(launch_path.read_text())
        command_args = launch["command"]
        target = int(command_args[command_args.index("--iterations") + 1])
        done = experiment / "verified.json"
        if done.exists():
            emit("SIGN_RESULT", json.loads(done.read_text()))
            continue
        result = verify(experiment / "metrics.jsonl", target, launch["run_id"], smoke=target == 2)
        result.update(name=experiment.name, lr=command_args[command_args.index("--lr") + 1],
                      size="257m" if "257m" in experiment.name else "500m",
                      metrics=str(experiment / "metrics.jsonl"), recovered_verification=True)
        atomic_json(done, result)
        emit("SIGN_RESULT", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["probe", "audit", "smoke", "tune", "long"])
    parser.add_argument("--queue", choices=["A", "B"])
    parser.add_argument("--lr", choices=["2e-3", "1e-3", "5e-4", "1e-4"], default="1e-4")
    parser.add_argument("--gpus", type=int, choices=[1, 2])
    parser.add_argument("--dataset", type=Path, default=Path("/workspace-SR006.nfs3/dimativator/fineweb-h200-packed"))
    parser.add_argument("--output", type=Path, default=Path("/home/jovyan/dimativator/sign-ademamix-20261007"))
    args = parser.parse_args()
    os.environ["WANDB_BASE_URL"] = "https://wandb-radfan.ru"
    os.environ["WANDB_ENTITY"] = "andrey"
    if args.mode == "probe":
        probe(args.dataset)
        return
    if args.mode == "audit":
        audit(args.output)
        return
    mpi_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    world = args.gpus or (2 if args.mode == "long" else 1)
    if mpi_size not in (1, world) or (args.mode == "tune" and world != 1):
        raise RuntimeError(f"Expected one MPI worker or {world} training ranks, got {mpi_size}")
    if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
        for query in [["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu", "--format=csv"],
                      ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"]]:
            subprocess.run(query, check=True)
    if args.mode == "smoke":
        run_one("257m", args.lr, 2, f"smoke{world}gpu", args.output, args.dataset, world, smoke=True)
        return
    metadata = json.loads((args.dataset / "packed_metadata.json").read_text())
    required = 157000 if args.mode == "long" else 39250
    if int(metadata["iterations"]) < required:
        raise RuntimeError(f"Dataset has {metadata['iterations']} iterations, needs {required}")
    if args.mode == "tune":
        if not args.queue:
            parser.error("--queue is required for tuning")
        for lr in LR_QUEUES[args.queue]:
            run_one("257m", lr, 39250, "tune1xc", args.output, args.dataset, world)
        return
    if shutil.disk_usage(args.output.parent).free < 9_000_000_000:
        raise RuntimeError("Need at least 9 GB checkpoint headroom")
    trunk = run_one("257m", args.lr, 157000, "full4xc", args.output, args.dataset, world,
                    milestones=(35325, 70650))
    for budget, checkpoint in [(1, 35325), (2, 70650)]:
        run_one("257m", args.lr, BUDGETS["257m"][budget], f"decay{budget}xc", args.output, args.dataset,
                world, resume=trunk / "ckpts" / str(checkpoint))
    if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
        cleanup_checkpoint(args.output, trunk)
    if shutil.disk_usage(args.output).free < 7_000_000_000:
        raise RuntimeError("Need at least 7 GB checkpoint headroom before 500M")
    trunk = run_one("500m", args.lr, 150914, "full2xc", args.output, args.dataset, world, milestones=(67911,))
    run_one("500m", args.lr, 75457, "decay1xc", args.output, args.dataset, world,
            resume=trunk / "ckpts/67911")
    if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
        cleanup_checkpoint(args.output, trunk)
        emit("SIGN_CAMPAIGN_COMPLETE", {"lr": args.lr, "checkpoints_remaining":
            [str(path) for path in args.output.rglob("*.pt")]})


if __name__ == "__main__":
    main()
