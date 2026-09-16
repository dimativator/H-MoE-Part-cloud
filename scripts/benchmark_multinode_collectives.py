#!/usr/bin/env python3
"""Measure node-local and cross-node NCCL all-gather on a 2x8 GPU allocation."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--message-mib", default="4,16,64")
    args = parser.parse_args()
    args.message_mib = [int(value) for value in args.message_mib.split(",")]
    return args


def benchmark_all_gather(
    *,
    group: dist.ProcessGroup,
    group_size: int,
    message_bytes: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[float, float]:
    element_size = torch.empty((), dtype=dtype).element_size()
    if message_bytes % element_size:
        raise ValueError("message size must be divisible by dtype element size")
    elements = message_bytes // element_size
    value = torch.full((elements,), dist.get_rank() % 251, dtype=dtype, device=device)
    gathered = torch.empty(elements * group_size, dtype=dtype, device=device)

    dist.barrier(group=group)
    for _ in range(warmup):
        dist.all_gather_into_tensor(gathered, value, group=group)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.barrier(group=group)
    start.record()
    for _ in range(repeats):
        dist.all_gather_into_tensor(gathered, value, group=group)
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end) / repeats
    maximum = torch.tensor(elapsed_ms, dtype=torch.float64, device=device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    maximum_ms = maximum.item()
    received_bytes = message_bytes * (group_size - 1)
    effective_gbps = received_bytes / maximum_ms / 1e6
    del value, gathered, maximum
    return maximum_ms, effective_gbps


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 16:
        raise RuntimeError(f"expected WORLD_SIZE=16, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    group_specs = {
        "local_node0": list(range(0, 8)),
        "local_node1": list(range(8, 16)),
        "cross_4plus4_a": [0, 1, 2, 3, 8, 9, 10, 11],
        "cross_4plus4_b": [4, 5, 6, 7, 12, 13, 14, 15],
    }
    groups = {name: dist.new_group(ranks) for name, ranks in group_specs.items()}
    local_name = "local_node0" if rank < 8 else "local_node1"
    cross_name = "cross_4plus4_a" if rank % 8 < 4 else "cross_4plus4_b"
    topology_groups = (
        ("intra_node_dp8", local_name, groups[local_name], 8),
        ("inter_node_dp8", cross_name, groups[cross_name], 8),
        ("inter_node_dp16", "world", dist.group.WORLD, 16),
    )
    dtypes = (("uint8", torch.uint8), ("bfloat16", torch.bfloat16))
    results = []
    for topology, group_name, group, group_size in topology_groups:
        leader = group_specs[group_name][0] if group_name != "world" else 0
        for dtype_name, dtype in dtypes:
            for message_mib in args.message_mib:
                elapsed_ms, effective_gbps = benchmark_all_gather(
                    group=group,
                    group_size=group_size,
                    message_bytes=message_mib * 2**20,
                    dtype=dtype,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    device=device,
                )
                if rank == leader:
                    results.append(
                        {
                            "topology": topology,
                            "group": group_name,
                            "group_size": group_size,
                            "dtype": dtype_name,
                            "message_mib_per_rank": message_mib,
                            "max_rank_ms": round(elapsed_ms, 4),
                            "effective_received_gbps": round(effective_gbps, 4),
                        }
                    )
                dist.barrier()
                torch.cuda.empty_cache()

    rank_info = {
        "rank": rank,
        "local_rank": local_rank,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(local_rank),
    }
    gathered_info: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered_info, rank_info)
    gathered_results: list[list[dict] | None] = [None] * world_size
    dist.all_gather_object(gathered_results, results)
    if rank == 0:
        payload = {
            "world_size": world_size,
            "ranks": gathered_info,
            "results": [row for rank_rows in gathered_results for row in rank_rows or []],
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "collectives.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print("MULTINODE_COLLECTIVE_RESULTS " + json.dumps(payload), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
