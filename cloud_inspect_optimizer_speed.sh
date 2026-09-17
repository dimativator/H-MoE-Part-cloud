#!/usr/bin/env bash
set -eu

output_root=${BENCHMARK_OUTPUT_DIR:?BENCHMARK_OUTPUT_DIR must point to a benchmark run or suite}

while IFS= read -r results; do
    echo "=== $results ==="
    cat "$results"
done < <(find "$output_root" -name results.csv -type f -print | sort)

while IFS= read -r log; do
    if grep -qE 'Traceback|ChildFailedError|ERROR|Error' "$log"; then
        echo "=== $log ==="
        tail -n 160 "$log"
    fi
done < <(find "$output_root" -path '*/logs/*.log' -type f -print | sort)
