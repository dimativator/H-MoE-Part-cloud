#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-full}
RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/frugal-muon-memory-cloud-20260917}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}

case "${MODE}" in
    smoke|full) ;;
    *) echo "MODE must be smoke or full" >&2; exit 2 ;;
esac
case "${RESULTS_DIR}" in
    /workspace-SR006.nfs3/dimativator/frugal-muon-memory-cloud-*) ;;
    *) echo "Unexpected RESULTS_DIR: ${RESULTS_DIR}" >&2; exit 2 ;;
esac

mkdir -p "${RESULTS_DIR}" "${EVAL_CACHE_DIR}" /tmp/triton-frugal-muon-memory
exec > >(tee -a "${RESULTS_DIR}/runner-${MODE}.log") 2>&1

echo "MODE=${MODE} RESULTS_DIR=${RESULTS_DIR}"
echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

python - <<'PY'
import torch
import torchao
import triton

assert torch.__version__.startswith("2.9.1"), torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith("12.8"), torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
assert torchao.__version__.startswith("0.15.0"), torchao.__version__
assert triton.__version__ == "3.5.1", triton.__version__
print("ENVIRONMENT_CHECK=ok", torch.__version__, torch.version.cuda)
PY

args=(
    --output-dir "${RESULTS_DIR}"
    --datasets-dir "${DATASETS_DIR}"
    --eval-cache-dir "${EVAL_CACHE_DIR}"
)
if [[ "${MODE}" == "smoke" ]]; then
    args+=(--smoke)
fi
python scripts/cloud/benchmark_frugal_muon_memory.py "${args[@]}"
