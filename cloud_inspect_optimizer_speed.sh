#!/usr/bin/env bash
set -eu

output_dir=${BENCHMARK_OUTPUT_DIR:?BENCHMARK_OUTPUT_DIR must point to a benchmark run}

if [[ -f "$output_dir/results.csv" ]]; then
    cat "$output_dir/results.csv"
fi

for log in "$output_dir"/logs/*.log; do
    [[ -f "$log" ]] || continue
    if grep -qE 'Traceback|ChildFailedError|ERROR|Error' "$log"; then
        echo "=== $log ==="
        tail -n 160 "$log"
    fi
done
