#!/usr/bin/env bash
set -eu

result_dir=${RESULT_DIR:?RESULT_DIR must point to one benchmark output directory}

echo "RESULT_DIR=$result_dir"

for result in "$result_dir"/results.csv "$result_dir"/results.json; do
    if [[ -f "$result" ]]; then
        echo "=== $result ==="
        cat "$result"
    fi
done

if [[ ${RESULTS_ONLY:-0} == 1 ]]; then
    exit 0
fi

find "$result_dir" -maxdepth 2 -type f -printf '%p\n' | sort

for log in "$result_dir"/logs/*.log; do
    [[ -f "$log" ]] || continue
    echo "=== $log ==="
    tail -n 240 "$log"
done
