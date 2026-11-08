"""Read campaign logs without touching training processes or checkpoint files."""
import json
from pathlib import Path
import os
import base64
import gzip

from rebuttal_storage_preflight import snapshot

root = Path(os.environ.get("REBUTTAL_INSPECT_ROOT", "/home/jovyan/rebuttal-memory-20261108"))
compact = {}
print("REBUTTAL_STORAGE=" + json.dumps(snapshot()), flush=True)
for path in sorted(root.glob("**/disk_guard.jsonl")):
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 10000))
        lines = handle.read().decode(errors="replace").splitlines()
    if lines:
        print("WORKER_GUARD=" + lines[-1], flush=True)
for path in sorted(root.glob("**/result.json")):
    result = json.loads(path.read_text())
    steps = [e for e in result["events"] if e["event"] == "step"]
    print("RESULT=" + json.dumps(dict(path=str(path), trial=result["trial"],
          status=result["status"], steps=len(steps))), flush=True)
    if os.environ.get("REBUTTAL_EXPORT_RESULTS") == "compact":
        for event in result["events"]:
            for value in event.get("inventory", {}).values():
                value.pop("storages", None)
        compact[str(path.relative_to(root))] = result
    if os.environ.get("REBUTTAL_EXPORT_RESULTS") == "1":
        payload = base64.b64encode(gzip.compress(path.read_bytes())).decode()
        for offset in range(0, len(payload), 12000):
            print("RESULT_EXPORT=" + json.dumps(dict(
                path=str(path.relative_to(root)), offset=offset,
                total=len(payload), data=payload[offset:offset + 12000])), flush=True)
for path in sorted(root.glob("**/train.log")):
    if "interrupted" in path.relative_to(root).parts:
        continue
    if (path.parent / "result.json").exists():
        continue
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 10000))
        tail = handle.read().decode(errors="replace")
    print("LIVE_LOG=" + json.dumps(dict(path=str(path), modified=path.stat().st_mtime,
                                      tail=tail)), flush=True)
if compact:
    payload = base64.b64encode(gzip.compress(json.dumps(compact).encode())).decode()
    for offset in range(0, len(payload), 12000):
        print("COMPACT_EXPORT=" + json.dumps(dict(offset=offset, total=len(payload),
              data=payload[offset:offset + 12000])), flush=True)
