#!/usr/bin/env bash
set -euo pipefail

METHOD=${METHOD:?METHOD must be galore or frugal}
case "${METHOD}" in
    galore)
        RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs2/dimativator/galore-rho-257m-1xc-wsd-20260914}
        ;;
    frugal)
        RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs2/dimativator/frugal-rho-257m-1xc-wsd-20260915}
        ;;
    *)
        echo "METHOD must be galore or frugal" >&2
        exit 2
        ;;
esac

rhos=(1p0 0p9 0p8 0p7 0p6 0p5 0p4 0p3 0p25 0p2 0p15 0p1)
done_count=0
for rho in "${rhos[@]}"; do
    marker="${RESULTS_DIR}/.llama257M_${METHOD}_rho${rho}_bf16_wsd_lr1e-3_wd0p1_1xC_1gpu.full.done"
    if [[ -f "${marker}" ]]; then
        echo "DONE rho=${rho} marker=${marker}"
        done_count=$((done_count + 1))
    else
        echo "MISSING rho=${rho} marker=${marker}"
    fi
done

echo "DONE_COUNT=${done_count} EXPECTED=${#rhos[@]} METHOD=${METHOD}"
