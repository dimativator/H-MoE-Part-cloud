"""Evaluate fixed FineWeb sequences on the live model and bootstrap the final mean."""
import base64
import hashlib
import json
import os
from pathlib import Path
import zlib

import numpy as np
import torch
from functools import lru_cache
import torch.nn.functional as F

DATA = Path("/workspace-SR006.nfs3/dimativator/fineweb-validation-32768-20261125")
DIGEST = "bc23cac94158ff98af16de2b31bf113752b0e5de1891786a27e9d279ef98c232"


def bootstrap(values):
    rng = np.random.default_rng(20260827)
    means = np.empty(100_000)
    for first in range(0, len(means), 64):
        count = min(64, len(means) - first)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[first:first + count] = values[indices].mean(axis=1)
    q = np.quantile(means, [0.005, 0.025, 0.975, 0.995])
    return dict(n=len(values), mean=float(values.mean()), low95=q[1], high95=q[2],
                width95=q[2]-q[1], low99=q[0], high99=q[3], width99=q[3]-q[0])


@lru_cache(maxsize=1)
def blocks():
    raw = (DATA / "validation.uint16").read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError("Validation digest mismatch")
    return np.frombuffer(raw, dtype="<u2").reshape(32768, 1025)


@torch.inference_mode()
def evaluate(model, reader, cfg, ctx, final):
    data = blocks()
    if not np.array_equal(reader.blocks.cpu().numpy(), data[:1024]):
        raise ValueError("Small validation differs from the shared 1024-sequence prefix")
    losses, accuracy = [], []
    for start in range(0, len(data), cfg.eval_batch_size):
        batch = torch.from_numpy(data[start:start+cfg.eval_batch_size].astype(np.int64)).to(cfg.device)
        y = batch[:, 1:]
        with ctx:
            logits = model(batch[:, :-1], targets=y, get_logits=True)["logits"]
            # Chunk CE to bound the temporary FP32 logits allocation.
            nll = [F.cross_entropy(l.flatten(0, 1), t.reshape(-1), reduction="none").reshape(t.shape).mean(1)
                   for l, t in zip(logits.split(4), y.split(4))]
        losses.append(torch.cat(nll).cpu().numpy())
        accuracy.append((logits.argmax(-1) == y).float().mean().item())
        del logits
    losses = np.concatenate(losses).astype("<f4")
    if not np.isfinite(losses).all():
        raise ValueError("Non-finite validation loss")
    mean = float(losses.mean(dtype=np.float64))
    metrics = {"val32768/loss": mean, "val1024/loss": float(losses[:1024].mean(dtype=np.float64))}
    if final:
        directory = Path(cfg.metrics_jsonl).parent
        np.save(directory / "per_sequence_losses.npy", losses)
        ci = bootstrap(losses.astype(np.float64))
        (directory / "bootstrap.json").write_text(json.dumps(ci))
        metrics.update({"val32768/low95": ci["low95"], "val32768/high95": ci["high95"]})
        if cfg.wandb:
            import wandb
            wandb.save(str(directory / "per_sequence_losses.npy"), base_path=str(directory), policy="now")
    return float(np.mean(accuracy)), mean, float(np.exp(mean)), metrics
