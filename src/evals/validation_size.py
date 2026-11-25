"""Evaluate the final 257M checkpoint using the Stage 3 sequence bootstrap."""
import base64
import hashlib
import json
import os
from pathlib import Path
import zlib

import numpy as np
import torch
import torch.distributed as dist
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


@torch.inference_mode()
def evaluate(model, exp_dir, cfg):
    dist.barrier()
    rank = dist.get_rank()
    path = exp_dir / "ckpts" / str(cfg.iterations) / "main.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint["itr"] != 39250:
        raise RuntimeError("Unexpected checkpoint iteration")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval()
    raw = (DATA / "validation.uint16").read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise RuntimeError("Validation checksum mismatch")
    blocks = np.frombuffer(raw, dtype="<u2").reshape(32768, 1025)
    local = np.array_split(blocks, dist.get_world_size())[rank]
    parts = []
    for first in range(0, len(local), 4):
        batch = torch.from_numpy(local[first:first+4].astype(np.int64)).to(cfg.device)
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, targets=y, get_logits=True)["logits"]
            per_token = F.cross_entropy(logits.flatten(0, 1), y.reshape(-1), reduction="none")
        parts.append(per_token.reshape(y.shape).mean(1).cpu().numpy())
    gathered = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(np.concatenate(parts), gathered, dst=0)
    if rank == 0:
        losses = np.concatenate(gathered)
        if losses.shape != (32768,) or not np.isfinite(losses).all():
            raise RuntimeError("Incomplete per-sequence results")
        np.save(exp_dir / "per_sequence_losses.npy", losses)
        rows = [bootstrap(losses[:n].astype(np.float64)) for n in (1024,2048,4096,8192,16384,32768)]
        result = dict(method=cfg.opt, validation_sha256=DIGEST, replicates=100000,
                      bootstrap_seed=20260827, intervals=rows)
        tmp = exp_dir / "bootstrap.tmp"
        tmp.write_text(json.dumps(result))
        tmp.replace(exp_dir / "bootstrap.json")
        encoded = base64.b64encode(zlib.compress(losses.astype("<f4").tobytes())).decode()
        for i in range(0, len(encoded), 2048):
            print("VAL_CI_LOSSES", i, encoded[i:i+2048], flush=True)
        print("VAL_CI_RESULT", json.dumps(result), flush=True)
    dist.barrier()
