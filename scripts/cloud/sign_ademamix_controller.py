#!/usr/bin/env python3
"""Run on brain_lab: submit two LR queues, select their best LR, submit long runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import netrc
from pathlib import Path
import re
import subprocess
import time


REPO = "https://github.com/dimativator/H-MoE-Part-cloud.git"
BRANCH = "codex/sign-ademamix-20261007"
ENTRY = "scripts/cloud/run_sign_ademamix_campaign.sh"


def save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def execute(arguments: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(arguments, capture_output=True, text=True, timeout=timeout)


def submit(path: Path, state: dict, key: str, gpus: int, arguments: str) -> None:
    if key in state:
        return
    if state.get("submission_pending"):
        raise RuntimeError("Unconfirmed prior submission. Audit scheduler before retrying.")
    active = execute(["timeout", "-k", "2", "15", "mlsub", "list", "--active"])
    active.check_returncode()
    note = f"Sign-AdEMAMix {key} 20261007"
    if note in active.stdout:
        raise RuntimeError("Matching active job exists but is missing from state")
    credential = netrc.netrc().authenticators("wandb-radfan.ru")
    if credential is None:
        raise RuntimeError("Missing existing W&B credential")
    command = ["mlsub", "run", "--repo", REPO, "--branch", BRANCH, "--entry", ENTRY,
        "--gpus", str(gpus), "--image", "efficient", "--no-pip", "--note", note,
        "--args", arguments, "--env", "WANDB_API_KEY=" + credential[2]]
    dry = execute(command + ["--dry-run"])
    if dry.returncode:
        raise RuntimeError("mlsub dry-run failed")
    state["submission_pending"] = {"key": key, "gpus": gpus, "arguments": arguments, "time": time.time()}
    save(path, state)
    result = execute(command)
    # Never print submission output, because it can contain credential arguments.
    jobs = re.findall(r"lm-mpi-job-[a-f0-9-]{36}", result.stdout)
    if result.returncode or not jobs:
        raise RuntimeError("Submission outcome uncertain. Inspect scheduler, do not resubmit.")
    state[key] = {"job": jobs[-1], "gpus": gpus, "arguments": arguments, "submitted_at": time.time()}
    state.pop("submission_pending")
    save(path, state)
    print("SIGN_SUBMITTED " + json.dumps(state[key]), flush=True)


def completed(path: Path, state: dict, key: str, expected_results: int) -> bool:
    job = state[key]
    if job.get("verified"):
        return True
    result = execute(["timeout", "-k", "2", "15", "mlsub", "status", job["job"]])
    if result.returncode:
        return False
    status = json.loads(result.stdout)
    job["scheduler"] = status["status"]
    job["checked_at"] = time.time()
    save(path, state)
    if status["status"].lower() in ("failed", "stopped", "cancelled"):
        raise RuntimeError(f"Job ended without success: {job['job']}")
    if status["status"].lower() != "completed":
        return False
    logs = execute(["timeout", "-k", "2", "15", "mlsub", "logs", job["job"]])
    if logs.returncode:
        return False
    (path.parent / f"{key}.log").write_text(logs.stdout)
    records = []
    completion = None
    exits = []
    for line in logs.stdout.splitlines():
        if "SIGN_RESULT " in line:
            records.append(json.JSONDecoder().raw_decode(line.split("SIGN_RESULT ", 1)[1])[0])
        if "SIGN_CAMPAIGN_COMPLETE " in line:
            completion = json.JSONDecoder().raw_decode(line.split("SIGN_CAMPAIGN_COMPLETE ", 1)[1])[0]
        if "SIGN_ENTRY_EXIT=" in line:
            exits.append(int(re.search(r"SIGN_ENTRY_EXIT=(\d+)", line).group(1)))
    if not exits or any(exits) or len(records) != expected_results:
        raise RuntimeError(f"Saved job logs fail completion checks: {job['job']}")
    if key == "long" and (completion is None or completion["checkpoints_remaining"]):
        raise RuntimeError("Long runs complete but checkpoint deletion is unverified")
    job.update(verified=True, results=records, completion=completion)
    save(path, state)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with (args.state.parent / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(args.state.read_text())
        if not state.get("smokes_verified"):
            raise RuntimeError("Require verified one-GPU and two-GPU smokes before tuning")
        try:
            submit(args.state, state, "queue_A", 1, "tune --queue A")
            submit(args.state, state, "queue_B", 1, "tune --queue B")
            while True:
                ready_A = completed(args.state, state, "queue_A", 2)
                ready_B = completed(args.state, state, "queue_B", 2)
                if ready_A and ready_B:
                    records = state["queue_A"]["results"] + state["queue_B"]["results"]
                    if {row["lr"] for row in records} != {"2e-3", "1e-3", "5e-4", "1e-4"}:
                        raise RuntimeError("Tuning grid incomplete")
                    best = min(records, key=lambda row: (row["final_val_loss"], float(row["lr"])))
                    state["best_lr"] = best["lr"]
                    state["selection"] = {"criterion": "minimum finite final validation loss at 39250 steps",
                                          "selected": best, "candidates": records}
                    save(args.state, state)
                    submit(args.state, state, "long", 2, "long --lr " + best["lr"])
                    if completed(args.state, state, "long", 5):
                        state["status"] = "complete"
                        save(args.state, state)
                        print("SIGN_CONTROLLER_COMPLETE", flush=True)
                        return
                state["status"] = "long_running" if "long" in state else "tuning_running"
                save(args.state, state)
                time.sleep(args.poll_seconds)
        except Exception as error:
            state["status"] = "needs_attention"
            state["error"] = str(error)
            save(args.state, state)
            raise


if __name__ == "__main__":
    main()
