#!/usr/bin/env bash
set -euo pipefail

if [[ ${OMPI_COMM_WORLD_RANK:-0} != 0 ]]; then
    exit 0
fi

root=$(cd "$(dirname "$0")" && pwd)
output=${DIAG_OUTPUT:-/home/jovyan/hmoe-cloud/adam-fp8-state-diagnosis.log}

if [[ ${DIAG_INSPECT_ONLY:-0} == 1 ]]; then
    python - "$output" <<'PY'
import re
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
variant = None
values = defaultdict(lambda: defaultdict(list))
step_means = {}
timer_re = re.compile(r"^\s+([a-z0-9-]+)\s+.*: \(([0-9.]+), [0-9.]+\)")
end_re = re.compile(r"DIAG_VARIANT_END=(\S+).*mean_step_ms=([0-9.]+)")
with open(path) as handle:
    for line in handle:
        if line.startswith("TRANSFORMER_ENGINE_VERSION=") or line.startswith(
            "FUSED_ADAM_SIGNATURE="
        ):
            print(line.rstrip())
        if line.startswith("DIAG_VARIANT_START="):
            variant = line.strip().split("=", 1)[1]
            continue
        match = end_re.search(line)
        if match:
            step_means[match.group(1)] = float(match.group(2))
            continue
        match = timer_re.match(line)
        if variant is not None and match:
            values[variant][match.group(1)].append(float(match.group(2)))

timers = (
    "forward-backward",
    "optimizer",
    "optimizer-copy-to-main-grad",
    "optimizer-clip-main-grad",
    "optimizer-inner-step",
    "optimizer-copy-main-to-model-params",
    "params-all-gather",
)
for name in values:
    fields = [f"variant={name}", f"step_ms={step_means.get(name)}"]
    for timer in timers:
        samples = values[name].get(timer, [])[-10:]
        mean = statistics.mean(samples) if samples else None
        fields.append(f"{timer}_ms={mean}")
    print("DIAG_SUMMARY " + " ".join(fields))
PY
    exit 0
fi

if [[ ${MLSUB_IMAGE:-} == torch28 ]]; then
    unset PYTHONNOUSERSITE
    nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
        -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
    export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
    export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
    export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
mkdir -p "$(dirname "$output")"
python "$root/scripts/diagnose_adam_fp8_states.py" 2>&1 | tee "$output"
