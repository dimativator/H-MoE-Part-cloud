"""Audit genuine two-rank FP8 checkpoints, optionally compare exact resume."""
import argparse
from pathlib import Path

import torch


def audit(directory, iteration):
    directory = Path(directory)
    main = torch.load(directory / "main.pt", map_location="cpu", mmap=True, weights_only=False)
    assert main["itr"] == iteration, main["itr"]
    assert main["scheduler"]["last_epoch"] == iteration
    assert main["model"] and main["optimizer"]["state"]
    state = next(iter(main["optimizer"]["state"].values()))
    assert "scale_exp_avg" in state and "scale_exp_avg_sq" in state
    workers = []
    for rank in range(2):
        worker = torch.load(directory / f"worker_{rank}.pt", map_location="cpu", weights_only=False)
        reader = worker["train_reader_state"]
        assert reader["reader_type"] == "packed_fineweb_train_reader_v1"
        assert reader["rank"] == rank and reader["batch_size"] == 16
        assert reader["sequence_length"] == 1024 and reader["step"] == iteration * 4, reader
        for key in ("rng_torch_cpu", "rng_torch_gpu", "rng_np", "rng_python"):
            assert key in worker, key
        workers.append(worker)
    print(f"CHECKPOINT_AUDIT_OK path={directory} itr={iteration} worker_steps={[w['train_reader_state']['step'] for w in workers]}", flush=True)
    return main, workers


def compare(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        assert left.shape == right.shape and left.dtype == right.dtype, path
        # Float8 CPU comparison is not supported in all torch builds.
        if left.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            left, right = left.float(), right.float()
        assert torch.equal(left, right), f"Resume mismatch: {path}"
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{path}.{index}")
    else:
        assert left == right, path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--compare")
    parser.add_argument("--numerical-next-step", action="store_true")
    args = parser.parse_args()
    main, workers = audit(args.directory, args.iteration)
    if args.compare:
        other, other_workers = audit(args.compare, args.iteration)
        if args.numerical_next_step:
            maximum = 0.0
            for key, value in main["model"].items():
                right = other["model"][key]
                difference = (value.float() - right.float()).abs()
                maximum = max(maximum, difference.max().item())
                # BF16 compute and DDP reductions need not be bitwise deterministic.
                torch.testing.assert_close(value, right, atol=2e-6, rtol=1e-5)
                assert difference.mean().item() < 2e-10, key
            compare(main["scheduler"], other["scheduler"])
            compare(main["optimizer"]["param_groups"], other["optimizer"]["param_groups"])
            print(f"NUMERICAL_NEXT_STEP_OK max_model_abs_diff={maximum}", flush=True)
        else:
            compare(main, other)
        for rank in range(2):
            compare(workers[rank]["train_reader_state"], other_workers[rank]["train_reader_state"])
        print("RESUME_CHECK_SCHEDULER_READERS_OK", flush=True)
