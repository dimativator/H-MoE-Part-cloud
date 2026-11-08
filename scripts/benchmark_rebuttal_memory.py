"""Isolated Megatron B1/B2/B4 trials, logs only, at most eight GPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from benchmark_training_step import build_command
from rebuttal_storage_preflight import snapshot

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("adam", "muon", "soap", "ademamix", "galore", "frugal",
           "frugal_muon_muon", "apollo", "slim_adam")
LOCK = threading.Lock()
STOP = threading.Event()
CHILDREN = {}


def grid(suite: str) -> list[dict]:
    if suite == "smoke":
        return [dict(suite=suite, method=m, states=s, ac=a, offload=o, mb=1,
                     dp=1, steps=2, trace=False, fused=False)
                for m, s, a, o in (("adam", "fp32", "none", False),
                                   ("muon", "fp8", "selective", False),
                                   ("frugal", "fp8", "full", True),
                                   ("slim_adam", "fp8", "none", False))]
    trials = []
    if suite in ("all", "b1"):
        for method, states, ac, offload, mb in itertools.product(
                METHODS, ("fp32", "fp8"), ("none", "selective", "full"),
                (False, True), (1, 2, 4, 8, 16, 32)):
            trials.append(dict(suite="b1", method=method, states=states, ac=ac,
                               offload=offload, mb=mb, dp=1,
                               steps=50 if mb == 1 else 15, trace=False, fused=False))
    if suite in ("all", "b2"):
        for method, mb in itertools.product(("muon", "adam", "frugal", "slim_adam", "galore"), (1, 32)):
            trials.append(dict(suite="b2", method=method, states="fp8" if method in ("muon", "adam") else "fp32",
                               ac="none", offload=False, mb=mb, dp=1,
                               steps=50 if method == "muon" else 15, trace=True, fused=False))
        for mb in (1, 32):
            trials.append(dict(suite="b2", method="muon", states="fp8", ac="none",
                               offload=False, mb=mb, dp=1, steps=50, trace=True, fused=True))
    if suite in ("all", "b4"):
        for method, dp, ac, mb in itertools.product(
                ("muon", "adam", "frugal", "slim_adam"), (1, 2, 4, 8),
                ("none", "full"), (1, 32)):
            trials.append(dict(suite="b4", method=method,
                               states="fp8" if method in ("muon", "adam") else "fp32",
                               ac=ac, offload=False, mb=mb, dp=dp, steps=15,
                               trace=False, fused=False))
    return trials


def trial_name(trial: dict) -> str:
    return "_".join(str(trial[k]).lower() for k in
                    ("suite", "method", "states", "ac", "offload", "mb", "dp", "fused"))


def command(trial: dict) -> list[str]:
    cmd = build_command(ROOT, "500m", "bf16", trial["method"],
                        trial["mb"] * trial["dp"], trial["mb"], trial["dp"],
                        1, trial["steps"] - 1, periodic_update_gap=5)
    index = cmd.index("--optimizer-state-precision") + 1
    cmd[index] = ("fp8_adam_fused" if trial["method"] == "adam" and trial["states"] == "fp8"
                  else trial["states"])
    cmd += ["--no-muon-distributed-state-sharding"]
    if trial["ac"] == "selective":
        cmd += ["--recompute-granularity", "selective", "--recompute-modules", "core_attn"]
    elif trial["ac"] == "full":
        cmd += ["--recompute-granularity", "full", "--recompute-method", "uniform",
                "--recompute-num-layers", "1"]
    return cmd


def guard(output: Path, extra_bytes: int = 0) -> None:
    record = snapshot()
    stat = os.statvfs(output)
    record["records"].append(dict(path=str(output), available_bytes=stat.f_bavail * stat.f_frsize,
                                  device=os.stat(output).st_dev, writable_by_campaign=True))
    failed = [r for r in record["records"] if r.get("writable_by_campaign", True)
              and ("error" in r or r["available_bytes"] <= 5_000_000_000 + extra_bytes)]
    record.update(status="stop" if failed else "ok", expected_write_bytes=extra_bytes)
    print("REBUTTAL_STORAGE=" + json.dumps(record), flush=True)
    if not failed:
        with LOCK, (output / "disk_guard.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
    if failed:
        STOP.set()
        with LOCK:
            children = list(CHILDREN.items())
        stopped = []
        for pid, child in children:
            try:
                os.killpg(pid, signal.SIGTERM)
                stopped.append(pid)
            except ProcessLookupError:
                pass
        for pid, child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
        print("CRITICAL CLOUD.RU STORAGE: EXPERIMENTS STOPPED. " +
              json.dumps(dict(failed=failed, stopped_process_groups=stopped,
                              checkpoints_available_for_cleanup=False)), flush=True)
        raise RuntimeError("Disk reserve unavailable; no checkpoints exist to delete")


def watchdog(output: Path) -> None:
    while not STOP.wait(30):
        try:
            guard(output)
        except Exception as error:
            STOP.set()
            print("GUARD_FAILURE=" + repr(error), flush=True)
            return


def run_trial(trial: dict, output: Path, devices: list[str]) -> dict:
    if STOP.is_set():
        raise RuntimeError("Campaign stopped by disk guard")
    guard(output, 100_000_000 if trial["trace"] else 10_000_000)
    run_dir = output / trial_name(trial)
    run_dir.mkdir(exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=",".join(devices), REBUTTAL_MEMORY="1",
               REBUTTAL_AC=trial["ac"], REBUTTAL_OFFLOAD=str(int(trial["offload"])),
               REBUTTAL_SHARD=str(int(trial["suite"] == "b4")),
               REBUTTAL_TRACE=str(int(trial["trace"])), REBUTTAL_RUN_DIR=str(run_dir),
               REBUTTAL_FUSED_FALLBACK=str(int(trial["fused"])),
               PYTHONUNBUFFERED="1", STAGE4_FP8_BATCHED="1", OMP_NUM_THREADS="1",
               WANDB_MODE="disabled", TORCHINDUCTOR_CACHE_DIR="/tmp/rebuttal-inductor",
               TRITON_CACHE_DIR="/tmp/rebuttal-triton-" + devices[0])
    env["PYTHONPATH"] = ":".join(str(p) for p in
        (ROOT / "third_party/Megatron-LM", ROOT / "third_party/emerging-optimizers", ROOT))
    cmd = command(trial)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (run_dir / "config.json").write_text(json.dumps(dict(trial=trial, command=cmd,
                                                       code_revision=revision), indent=2))
    log_path = run_dir / "train.log"
    print("TRIAL_START=" + json.dumps(dict(trial=trial, devices=devices)), flush=True)
    with log_path.open("w") as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        with LOCK:
            CHILDREN[child.pid] = child
        code = child.wait()
        with LOCK:
            CHILDREN.pop(child.pid, None)
    text = log_path.read_text(errors="replace")
    events = []
    for rank_log in sorted(run_dir.glob("rank*.jsonl")):
        for line in rank_log.read_text().splitlines():
            events.append(json.loads(line))
    steps = [event for event in events if event["event"] == "step"]
    complete = (code == 0 and all(sum(s["rank"] == rank for s in steps) == trial["steps"]
                                 for rank in range(trial["dp"]))
                and all(not s["skipped_iteration"] and
                        all(math.isfinite(v) for v in s["losses"].values()) for s in steps))
    nonfinite = any(not math.isfinite(v) for s in steps for v in s["losses"].values())
    status = ("success" if complete else "oom" if "out of memory" in text.lower()
              else "diverged" if nonfinite else "failed")
    result = dict(trial=trial, status=status, return_code=code, log=str(log_path), code_revision=revision,
                  events=events, timestamp=time.time())
    result_path.write_text(json.dumps(result, indent=2))
    print("TRIAL_END=" + json.dumps(dict(trial=trial, status=status, return_code=code)), flush=True)
    if status == "failed":
        print(text[-8000:], flush=True)
        STOP.set()
        raise RuntimeError("Trial failed: " + trial_name(trial))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "all", "b1", "b2", "b4"), default="smoke")
    parser.add_argument("--gpus", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--output", type=Path, default=Path("/home/jovyan/rebuttal-memory-20261108"))
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    trials = grid(args.suite)
    if args.plan:
        print(json.dumps([dict(trial=t, command=command(t)) for t in trials], indent=2))
        return
    if max(t["dp"] for t in trials) > args.gpus:
        parser.error("The selected suite requires eight GPUs")
    args.output.mkdir(parents=True, exist_ok=True)
    guard(args.output, 500_000_000)
    threading.Thread(target=watchdog, args=(args.output,), daemon=True).start()
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(args.gpus)))).split(",")
    if len(devices) < args.gpus:
        raise RuntimeError("Fewer visible GPUs than requested")
    import torch
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError("Scheduler allocation exposes fewer GPUs than requested")
    def run_singles(selected: list[dict], output: Path) -> list[dict]:
        lanes = [selected[i::args.gpus] for i in range(args.gpus)]
        def lane(index: int):
            return [run_trial(t, output, [devices[index]]) for t in lanes[index]]
        with ThreadPoolExecutor(max_workers=args.gpus) as pool:
            return [r for lane_results in pool.map(lane, range(args.gpus)) for r in lane_results]

    if args.suite == "all":
        validation_dir = args.output / "validation"
        validation_dir.mkdir(exist_ok=True)
        validation = [dict(suite="validation", method=m, states=s, ac="none", offload=False,
                           mb=1, dp=1, steps=2, trace=m == "muon" and s == "fp8", fused=False)
                      for m, s in itertools.product(METHODS, ("fp32", "fp8"))]
        validation.append(dict(suite="validation", method="muon", states="fp8", ac="none",
                               offload=False, mb=1, dp=1, steps=2, trace=True, fused=True))
        validation_results = run_singles(validation, validation_dir)
        for method in ("muon", "adam", "frugal", "slim_adam"):
            trial = dict(suite="b4", method=method, states="fp8" if method in ("muon", "adam") else "fp32",
                         ac="none", offload=False, mb=1, dp=2, steps=2, trace=False, fused=False)
            validation_results.append(run_trial(trial, validation_dir, devices[:2]))
        if any(r["status"] != "success" for r in validation_results):
            raise RuntimeError("Validation did not pass for every implementation")
        print("GPU_VALIDATION_COMPLETE", flush=True)

    results = run_singles([t for t in trials if t["suite"] in ("smoke", "b2")], args.output)
    for trial in trials:
        if trial["suite"] == "b4":
            results.append(run_trial(trial, args.output, devices[:trial["dp"]]))
    results += run_singles([t for t in trials if t["suite"] == "b1"], args.output)
    STOP.set()
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    subprocess.run([sys.executable, str(ROOT / "scripts/report_rebuttal_memory.py"),
                    str(args.output), str(args.output / "tables")], check=True)
    print("CAMPAIGN_COMPLETE=" + str(len(results)), flush=True)


if __name__ == "__main__":
    if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
        main()
