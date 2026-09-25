#!/usr/bin/env bash
set -euo pipefail

readonly target=/home/jovyan/finewebedu_h200/sample/100BT
readonly packed=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed

if [[ ! -d "${target}" || -L "${target}" || -L "$(dirname -- "${target}")" ]]; then
    echo "Refusing missing or symbolic-link target: ${target}" >&2
    exit 2
fi
if [[ "$(realpath -e -- "${target}")" != "${target}" ]]; then
    echo "Refusing non-canonical target: ${target}" >&2
    exit 2
fi
if [[ ! -f "${packed}/packed_metadata.json" ]]; then
    echo "Protected packed FineWeb is missing: ${packed}" >&2
    exit 3
fi
if [[ "$(stat -c %d -- "${target}")" == "$(stat -c %d -- "${packed}")" ]]; then
    echo "Target and protected packed data unexpectedly share a filesystem" >&2
    exit 3
fi

mapfile -t files < <(find "${target}" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)
expected=(000_00000.parquet 000_00001.parquet 000_00002.parquet 000_00003.parquet)
if [[ "${files[*]}" != "${expected[*]}" ]]; then
    echo "Refusing unexpected target contents: ${files[*]}" >&2
    exit 4
fi
for name in "${expected[@]}"; do
    if [[ ! -f "${target}/${name}" || -L "${target}/${name}" ]]; then
        echo "Refusing non-regular parquet file: ${target}/${name}" >&2
        exit 4
    fi
done

echo "BEFORE"
df -h /home/jovyan "${packed}"
du -sh -- "${target}"
echo "REMOVING=${target}"
rm -rf --one-file-system -- "${target}"
test ! -e "${target}"
test -f "${packed}/packed_metadata.json"
echo "REMOVED=${target}"
echo "PROTECTED=${packed}"
echo "AFTER"
df -h /home/jovyan "${packed}"
