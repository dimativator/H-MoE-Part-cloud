# Activation memory: checkpointing, offload and peak-memory profiling

Branch `act-checkpointing`, cut from `codex/500m-bf16-optimizer-wd-tuning-20261006`. It measures how activation checkpointing (AC) and activation offload (AO) change the peak-memory comparison between optimizers (Tables 1, 3, 9, 10 and 21 of the paper).

## Training flags (`src/config/base.py`)

| Flag | Effect |
|---|---|
| `--activation-checkpointing none\|mlp\|full` | `full`: every block runs under non-reentrant `torch.utils.checkpoint`, so only block inputs are kept and the block is recomputed in backward. `mlp`: only norm 2 + SwiGLU + residual add are recomputed. Both modes cover the BF16 and FP8 (COAT) block paths, apply only in training, and preserve RNG and autocast state. |
| `--activation-offload` | Tensors saved for backward go to pinned CPU memory (`torch.autograd.graph.save_on_cpu`). |
| `--mem-profile PATH` | Writes the per-phase memory JSON and exits after `--mem-profile-steps` (default 3) optimizer steps. |
| `--mem-snapshot PATH` | With `--mem-profile`: dumps a `torch.cuda.memory` snapshot of the last profiled step (view at https://pytorch.org/memory_viz). |

Recomputation does not change the numerics. On CPU, loss and gradients match with and without AC/AO (`tests/test_activation_memory.py`).

## What the profile records (`src/optim/mem_profile.py`)

Per step:
- allocated memory at step start;
- memory held after the first microbatch forward (saved activations + logits);
- `peak_fwd_bwd`: the peak over all microbatches' forward/backward plus gradient clipping;
- `peak_opt_step`: the peak inside `opt.step()`;
- `binding_phase`: which of the two sets the step peak;
- reserved memory and step time;
- persistent bytes split into parameters, gradients, optimizer state (walked from `opt.state`, including wrapped optimizers) and other model tensors (FP8 weight caches, buffers). Each storage is counted once.

The summary has two peaks:
- `peak_gb`: over all profiled steps. Step 0 includes lazy state initialisation and the first projection refresh.
- `peak_steady_gb`: steps ≥ 1 only.

## Sweep and tables

```bash
# 500M, BF16 compute, Table 1/3 grid under three AC modes; data args after "--"
python scripts/memory/sweep.py --out-dir results/mem500m --ac none,mlp,full -- \
    --dataset fineweb --datasets-dir /path/to/fineweb-packed \
    --fineweb-replay-world-size 1 --fineweb-replay-layout concat \
    --eval-cache-dir /path/to/evals_cache --streaming --workers 8

python scripts/memory/collect.py results/mem500m             # summary.md + summary.csv
python scripts/memory/collect.py results/mem500m --metric peak_steady_gb
```

Sweep defaults:
- all 9 optimizers, FP32 and FP8 states, BF16 compute;
- microbatch 1–32, with accumulation steps = 128 / microbatch;
- 3 profiled steps per case.

Other options: `--compute bf16,fp8` adds W8A8G8, `--offload off,on` adds AO, `--snapshot` saves memory snapshots, and `--dry-run` prints the commands. Finished cases are skipped on rerun. OOM cases are stored as results; other failures are retried.

## Caveats

- **Only tested on CPU.** The BF16 path is tested there. The FP8 (COAT) path and every CUDA number still need a GPU run. Before trusting FP8 + AC, check that a few steps give the same loss with `--activation-checkpointing none` and `full`.
- **Offload is synchronous.** Its memory numbers are valid, but its step time is pessimistic compared with an overlapped implementation. Only tensors saved through autograd are offloaded; anything a custom autograd function stores on `ctx` directly stays on the GPU. With AC, the checkpointed block inputs also stay on the GPU.
- **FP8 states are wired here only for AdamW (`triton_coat_adamw`), Muon, SOAP, AdEMAMix, SlimAdam and FRUGAL MM.** The sweep skips FP8-state GaLore, FRUGAL and APOLLO. Those live on the `galore-fp8`, `frugal-fp8`, `apollo-fp8` and `eco-nsight-bench` branches.
- **The train log's `peak_mem` is partial while profiling.** With `--mem-profile`, it covers only the optimizer step onward, because the profiler resets the peak counter before `opt.step()`. Use the JSON.
- **Eval runs once at iteration 0.** It is freed before the first profiled step, so it does not enter the peaks, but it can raise `reserved_gb`.
