#!/usr/bin/env bash
set -u
if [[ ${OMPI_COMM_WORLD_RANK:-0} != 0 ]]; then exit 0; fi
root=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
export OMP_NUM_THREADS=1
if [[ ${REBUTTAL_CPU_TEST:-0} == 1 ]]; then
    python "$root/scripts/rebuttal_storage_preflight.py"
    python -m torch.distributed.run --standalone --nproc-per-node 2 "$root/scripts/test_rebuttal_sharding.py"
    code=$?
    echo "REBUTTAL_EXIT=$code"
    exit 0
fi
python "$root/scripts/benchmark_rebuttal_memory.py" "$@"
code=$?
echo "REBUTTAL_EXIT=$code"
exit 0
