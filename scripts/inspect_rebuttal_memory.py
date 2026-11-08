"""Read campaign logs without touching training processes or checkpoint files."""
import json
from pathlib import Path
import os

from rebuttal_storage_preflight import snapshot

root = Path(os.environ.get("REBUTTAL_INSPECT_ROOT", "/home/jovyan/rebuttal-memory-20261108"))
print("REBUTTAL_STORAGE=" + json.dumps(snapshot()), flush=True)
for path in sorted(root.glob("**/result.json")):
    result = json.loads(path.read_text())
    steps = [e for e in result["events"] if e["event"] == "step"]
    print("RESULT=" + json.dumps(dict(path=str(path), trial=result["trial"],
          status=result["status"], steps=len(steps))), flush=True)
for path in sorted(root.glob("**/train.log")):
    if (path.parent / "result.json").exists():
        continue
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 10000))
        tail = handle.read().decode(errors="replace")
    print("LIVE_LOG=" + json.dumps(dict(path=str(path), modified=path.stat().st_mtime,
                                      tail=tail)), flush=True)
