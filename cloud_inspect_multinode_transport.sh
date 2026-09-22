#!/usr/bin/env bash
set -euo pipefail

output_root=${BENCHMARK_OUTPUT_DIR:?BENCHMARK_OUTPUT_DIR must point to a multinode benchmark root}

for artifact in \
    "$output_root"/topology/node*.txt \
    "$output_root"/preflight/collectives.json; do
    [[ -f "$artifact" ]] || continue
    echo "=== $artifact ==="
    cat "$artifact"
done

for log in "$output_root"/node-logs/*-node0.log "$output_root"/node-logs/*-node1.log; do
    [[ -f "$log" ]] || continue
    matches=$(grep -E \
        'NCCL INFO (Using network|NET/|Assigned NET plugin|Bootstrap)|NCCL_(IB|SOCKET)|libnccl-net|InfiniBand|RoCE|mlx[0-9]|Socket NIC' \
        -m 200 "$log" || true)
    [[ -n "$matches" ]] || continue
    echo "=== $log ==="
    printf '%s\n' "$matches" | sort -u
done
