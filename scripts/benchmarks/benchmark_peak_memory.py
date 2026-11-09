#!/usr/bin/env python3
"""Benchmark single-GPU peak allocated memory across optimizers and FP8 modes.

Each measurement is an isolated 10-step torchrun process. Batch sizes are doubled
from 1 until the first CUDA OOM. Results are appended to CSV after every attempt,
so an interrupted sweep can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import time


MODELS = {
    "257M": {"n_layer": 12, "n_embd": 1024, "n_head": 8},
    "500M": {"n_layer": 18, "n_embd": 1280, "n_head": 20},
}

OPTIMIZERS = {
    "AdamW": {"cli": "adamw", "fp8_cli": "triton_coat_adamw", "fp8_states": True},
    "Muon": {"cli": "muon", "fp8_states": True},
    "SOAP": {"cli": "soap", "fp8_states": True, "expensive": "update_gap=5"},
    "AdEMAMix": {"cli": "ademamix", "fp8_states": True},
    "Lion": {"cli": "lion", "fp8_states": True},
    "GaLore": {"cli": "galore_adamw", "fp8_states": True, "expensive": "update_gap=5"},
    "FRUGAL": {"cli": "coord_adamw", "fp8_states": True, "expensive": "update_gap=5"},
    "APOLLO": {"cli": "apollo_adamw", "fp8_states": True, "expensive": "update_gap=5"},
    "Slim Adam": {"cli": "slim_adam", "fp8_states": True},
    "Adam mini": {"cli": "adam_mini", "fp8_states": True},
    "Fira": {"cli": "fira_adamw", "fp8_states": True, "expensive": "update_gap=5"},
}

PRECISIONS = ("bf16", "fp8_act", "fp8_states", "fp8_full")
CSV_FIELDS = (
    "timestamp_utc",
    "model",
    "parameter_count_m",
    "optimizer",
    "optimizer_cli",
    "precision",
    "batch_size",
    "gpu",
    "status",
    "peak_memory_gb",
    "gpu_used_before_mib",
    "gpu_free_before_mib",
    "duration_sec",
    "return_code",
    "expensive_step_policy",
    "log_path",
    "note",
)

OOM_RE = re.compile(
    r"CUDA out of memory|torch\.OutOfMemoryError|out of memory|CUBLAS_STATUS_ALLOC_FAILED",
    re.IGNORECASE,
)
PEAK_RE = re.compile(r"peak_mem=([0-9.]+)GB")
PARAM_RE = re.compile(r"number of parameters:\s*([0-9.]+)M")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/data/users/dimativator/Efficient-Training"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/users/dimativator/Efficient-Training/benchmarks/"
            "peak_memory_all_optimizers_20260906"
        ),
    )
    parser.add_argument("--preferred-gpu", type=int, default=7)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def gpu_snapshot() -> dict[int, dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = {}
    for line in output.strip().splitlines():
        index, used, free, util = (int(value) for value in line.split(", "))
        result[index] = {"used": used, "free": free, "util": util}
    return result


def choose_gpu(preferred_gpu: int) -> tuple[int | None, dict[int, dict[str, int]]]:
    snapshot = gpu_snapshot()
    # The user's rule: GPU 7 is usable while all other allocations total < 40 GiB.
    if snapshot[preferred_gpu]["used"] < 40 * 1024:
        return preferred_gpu, snapshot

    candidates = [
        index
        for index, stats in snapshot.items()
        if index != preferred_gpu and stats["free"] >= 100 * 1024
    ]
    if not candidates:
        return None, snapshot
    return max(candidates, key=lambda index: snapshot[index]["free"]), snapshot


def precision_flags(precision: str) -> list[str]:
    activation_flags = [
        "--fp8",
        "--fp8-fabit",
        "E4M3",
        "--fp8-fwbit",
        "E4M3",
        "--fp8-babit",
        "E5M2",
        "--fp8-bwbit",
        "E5M2",
        "--fp8-group-size",
        "16",
    ]
    state_flags = [
        "--fp8-optim",
        "--fp8-qgroup-size",
        "128",
        "--fp8-first-order-bit",
        "E4M3",
        "--fp8-second-order-bit",
        "E4M3",
        "--fp8-expansion",
        "expand",
    ]
    if precision == "bf16":
        return []
    if precision == "fp8_act":
        return activation_flags
    if precision == "fp8_states":
        return state_flags
    if precision == "fp8_full":
        return activation_flags + state_flags
    raise ValueError(f"Unknown precision: {precision}")


def optimizer_cli(optimizer: str, precision: str) -> str:
    config = OPTIMIZERS[optimizer]
    if optimizer == "AdamW" and precision in ("fp8_states", "fp8_full"):
        return config["fp8_cli"]
    return config["cli"]


def command_for(
    args: argparse.Namespace,
    model: str,
    optimizer: str,
    precision: str,
    batch_size: int,
) -> list[str]:
    shape = MODELS[model]
    exp_name = (
        f"memory_bench_{model.lower()}_{OPTIMIZERS[optimizer]['cli']}_"
        f"{precision}_bs{batch_size}"
    )
    command = [
        "/data/users/dimativator/anaconda3/envs/eff-pretrain/bin/torchrun",
        "--standalone",
        "--nproc-per-node=1",
        "src/main.py",
        "--distributed-backend",
        "nccl",
        "--experiment-name",
        exp_name,
        "--dataset",
        "fineweb",
        "--datasets-dir",
        "/data/datasets/fineweb-edu-100BT/sample/100BT",
        "--eval-cache-dir",
        "/data/users/dimativator/evals_cache",
        "--sequence-length",
        "1024",
        "--streaming",
        "--workers",
        "2",
        "--model",
        "llama",
        "--n-layer",
        str(shape["n_layer"]),
        "--n-embd",
        str(shape["n_embd"]),
        "--n-head",
        str(shape["n_head"]),
        "--multiple-of",
        "256",
        "--dtype",
        "bfloat16",
        "--opt",
        optimizer_cli(optimizer, precision),
        "--lr",
        "1e-3",
        "--weight-decay",
        "1e-4",
        "--beta1",
        "0.9",
        "--beta2",
        "0.99",
        "--grad-clip",
        "1.0",
        "--scheduler",
        "none",
        "--warmup-steps",
        "0",
        "--iterations",
        str(args.steps),
        "--batch-size",
        str(batch_size),
        "--eval-batch-size",
        str(batch_size),
        "--acc-steps",
        "1",
        "--update_gap",
        "5",
        "--eval-interval",
        "1000000",
        "--eval-batches",
        "1",
        "--log-interval",
        "1",
        "--no-local-save",
    ]
    return command + precision_flags(precision)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(csv_path: Path, row: dict[str, object]) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def completed_attempts(rows: list[dict[str, str]]) -> set[tuple[str, str, str, int]]:
    done = set()
    for row in rows:
        if row["status"] in {"success", "oom"} and row["batch_size"]:
            done.add(
                (
                    row["model"],
                    row["optimizer"],
                    row["precision"],
                    int(row["batch_size"]),
                )
            )
    return done


def remove_resolved_unsupported(
    csv_path: Path, rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Drop obsolete markers after FP8 state support is implemented."""
    filtered = [
        row
        for row in rows
        if not (
            row["status"] == "unsupported"
            and OPTIMIZERS[row["optimizer"]]["fp8_states"]
        )
    ]
    if len(filtered) == len(rows):
        return rows

    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(filtered)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(csv_path)
    return filtered


def record_unsupported(csv_path: Path, rows: list[dict[str, str]]) -> None:
    existing = {
        (row["model"], row["optimizer"], row["precision"])
        for row in rows
        if row["status"] == "unsupported"
    }
    for model in MODELS:
        for optimizer, config in OPTIMIZERS.items():
            if config["fp8_states"]:
                continue
            for precision in ("fp8_states", "fp8_full"):
                if (model, optimizer, precision) in existing:
                    continue
                append_row(
                    csv_path,
                    {
                        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "model": model,
                        "parameter_count_m": "",
                        "optimizer": optimizer,
                        "optimizer_cli": config["cli"],
                        "precision": precision,
                        "batch_size": "",
                        "gpu": "",
                        "status": "unsupported",
                        "peak_memory_gb": "",
                        "gpu_used_before_mib": "",
                        "gpu_free_before_mib": "",
                        "duration_sec": "",
                        "return_code": "",
                        "expensive_step_policy": config.get("expensive", "every step"),
                        "log_path": "",
                        "note": "Optimizer states are not quantized by the current implementation.",
                    },
                )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    csv_path = args.output_dir / "results.csv"
    state_path = args.output_dir / "state.json"

    rows = remove_resolved_unsupported(csv_path, read_rows(csv_path))
    record_unsupported(csv_path, rows)
    rows = read_rows(csv_path)
    done = completed_attempts(rows)

    models = ["257M"] if args.smoke else list(MODELS)
    optimizers = ["AdamW"] if args.smoke else list(OPTIMIZERS)
    precisions = ["bf16"] if args.smoke else list(PRECISIONS)

    for model in models:
        for optimizer in optimizers:
            for precision in precisions:
                if precision in ("fp8_states", "fp8_full") and not OPTIMIZERS[optimizer]["fp8_states"]:
                    continue

                batch_size = 1
                while True:
                    key = (model, optimizer, precision, batch_size)
                    previous = [
                        row
                        for row in rows
                        if row["model"] == model
                        and row["optimizer"] == optimizer
                        and row["precision"] == precision
                        and row["batch_size"] == str(batch_size)
                    ]
                    if previous and previous[-1]["status"] == "oom":
                        break
                    if key in done:
                        batch_size *= 2
                        continue

                    gpu, snapshot = choose_gpu(args.preferred_gpu)
                    while gpu is None:
                        state_path.write_text(
                            json.dumps(
                                {
                                    "status": "waiting_for_gpu",
                                    "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                                    "gpu_snapshot": snapshot,
                                    "next": dict(
                                        model=model,
                                        optimizer=optimizer,
                                        precision=precision,
                                        batch_size=batch_size,
                                    ),
                                },
                                indent=2,
                            )
                        )
                        time.sleep(300)
                        gpu, snapshot = choose_gpu(args.preferred_gpu)

                    before = snapshot[gpu]
                    log_path = logs_dir / f"{model}_{optimizer}_{precision}_bs{batch_size}.log"
                    command = command_for(args, model, optimizer, precision, batch_size)
                    state_path.write_text(
                        json.dumps(
                            {
                                "status": "running",
                                "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                                "gpu": gpu,
                                "model": model,
                                "optimizer": optimizer,
                                "precision": precision,
                                "batch_size": batch_size,
                                "command": command,
                            },
                            indent=2,
                        )
                    )
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    env["PYTHONUNBUFFERED"] = "1"
                    started = time.monotonic()
                    timed_out = False
                    with log_path.open("w") as log:
                        log.write("COMMAND: " + " ".join(command) + "\n")
                        log.flush()
                        try:
                            result = subprocess.run(
                                command,
                                cwd=args.repo,
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=args.timeout_sec,
                            )
                            return_code = result.returncode
                        except subprocess.TimeoutExpired:
                            timed_out = True
                            return_code = 124

                    duration = time.monotonic() - started
                    text = log_path.read_text(errors="replace")
                    peaks = [float(value) for value in PEAK_RE.findall(text)]
                    params = PARAM_RE.search(text)
                    oom = bool(OOM_RE.search(text))
                    if timed_out:
                        status = "timeout"
                    elif oom:
                        status = "oom"
                    elif return_code == 0 and len(peaks) >= args.steps:
                        status = "success"
                    else:
                        status = "error"

                    note = ""
                    if status == "success":
                        note = f"Captured {len(peaks)} optimizer steps."
                    elif status == "oom":
                        note = "First CUDA OOM; larger batches skipped."
                    elif status == "timeout":
                        note = f"Exceeded {args.timeout_sec} seconds."
                    else:
                        note = "Non-OOM failure; inspect log before resuming this configuration."

                    row = {
                        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "model": model,
                        "parameter_count_m": params.group(1) if params else "",
                        "optimizer": optimizer,
                        "optimizer_cli": optimizer_cli(optimizer, precision),
                        "precision": precision,
                        "batch_size": batch_size,
                        "gpu": gpu,
                        "status": status,
                        "peak_memory_gb": max(peaks) if peaks else "",
                        "gpu_used_before_mib": before["used"],
                        "gpu_free_before_mib": before["free"],
                        "duration_sec": round(duration, 2),
                        "return_code": return_code,
                        "expensive_step_policy": OPTIMIZERS[optimizer].get("expensive", "every step"),
                        "log_path": str(log_path),
                        "note": note,
                    }
                    append_row(csv_path, row)
                    rows.append({key: str(value) for key, value in row.items()})
                    done.add(key)
                    if status in {"timeout", "error"}:
                        state_path.write_text(
                            json.dumps(
                                {
                                    "status": status,
                                    "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                                    "model": model,
                                    "optimizer": optimizer,
                                    "precision": precision,
                                    "batch_size": batch_size,
                                    "log_path": str(log_path),
                                },
                                indent=2,
                            )
                        )
                        return 1
                    if status == "oom":
                        break
                    if args.smoke:
                        state_path.write_text(json.dumps({"status": "smoke_complete"}, indent=2))
                        return 0
                    batch_size *= 2

    state_path.write_text(
        json.dumps(
            {"status": "complete", "timestamp_utc": dt.datetime.now(dt.UTC).isoformat()},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
