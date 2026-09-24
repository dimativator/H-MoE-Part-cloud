#!/usr/bin/env python3
"""Compare Adam execution paths with Megatron's built-in phase timers."""

import inspect
import os
import statistics
import subprocess
import sys
from pathlib import Path

from scripts.benchmark_training_step import ITERATION_RE, build_command


WARMUP_STEPS = 5
MEASURE_STEPS = 10
VARIANTS = {
    "adam": (),
    "adam_distributed_fp32": ("--use-distributed-optimizer",),
    "adam_precision_aware_fp32": (
        "--use-distributed-optimizer",
        "--use-precision-aware-optimizer",
        "--exp-avg-dtype",
        "fp32",
        "--exp-avg-sq-dtype",
        "fp32",
    ),
    "adam_precision_aware_fp8": (
        "--use-distributed-optimizer",
        "--use-precision-aware-optimizer",
        "--exp-avg-dtype",
        "fp8",
        "--exp-avg-sq-dtype",
        "fp8",
    ),
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    from transformer_engine import __version__ as transformer_engine_version
    from transformer_engine.pytorch.optimizers import FusedAdam

    print(f"TRANSFORMER_ENGINE_VERSION={transformer_engine_version}", flush=True)
    print(f"FUSED_ADAM_SIGNATURE={inspect.signature(FusedAdam)}", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(root / "third_party" / "Megatron-LM"),
            str(root / "third_party" / "emerging-optimizers"),
            str(root),
        ]
    )
    env["PYTHONUNBUFFERED"] = "1"
    failures = 0
    for variant, extra_args in VARIANTS.items():
        command = build_command(
            root=root,
            model_name="1b",
            precision="bf16",
            optimizer="adam",
            global_batch=1,
            micro_batch=1,
            data_parallel_size=1,
            warmup=WARMUP_STEPS,
            measured=MEASURE_STEPS,
        )
        command.extend(("--timing-log-level", "1", "--logging-level", "20"))
        command.extend(extra_args)
        print(f"DIAG_VARIANT_START={variant}", flush=True)
        print("DIAG_COMMAND=" + " ".join(command), flush=True)
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output_lines = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output_lines.append(line)
        return_code = process.wait()
        output = "".join(output_lines)
        times = [
            float(match.group(2))
            for match in ITERATION_RE.finditer(output)
            if int(match.group(1)) > WARMUP_STEPS
        ][:MEASURE_STEPS]
        mean_ms = statistics.mean(times) if times else None
        print(
            f"DIAG_VARIANT_END={variant} return_code={return_code} "
            f"samples={len(times)} mean_step_ms={mean_ms}",
            flush=True,
        )
        failures += return_code != 0 or len(times) != MEASURE_STEPS
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
