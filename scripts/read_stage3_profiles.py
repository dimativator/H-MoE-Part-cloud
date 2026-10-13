"""Read existing Stage III CPU/CUDA profiles without running training."""

import argparse
import base64
from collections import defaultdict
import gzip
import json
from pathlib import Path


def emit(kind: str, name: str, payload: object) -> None:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    encoded = base64.b64encode(gzip.compress(raw)).decode()
    for index in range(0, len(encoded), 12000):
        print(f"STAGE3_DATA {kind} {name} {index // 12000} {encoded[index:index+12000]}", flush=True)
    print(f"STAGE3_END {kind} {name}", flush=True)


def union_us(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def trace_summary(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        trace = json.load(handle)
    events = trace["traceEvents"]
    ranges = [e for e in events if e.get("ph") == "X" and e.get("name", "").startswith("stage3::")]
    cpu = {e["args"]["External id"]: e for e in events
           if e.get("ph") == "X" and e.get("cat") == "cpu_op" and "External id" in e.get("args", {})}
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    totals = defaultdict(lambda: [0, 0.0])
    mapped = defaultdict(lambda: [0, 0.0])
    phase_intervals = defaultdict(list)
    for event in kernels:
        name, duration = event["name"], event["dur"]
        totals[name][0] += 1
        totals[name][1] += duration
        parent = cpu.get(event.get("args", {}).get("External id"))
        phase = "unmapped"
        owner = ""
        if parent is not None:
            matches = [r for r in ranges if r["ts"] <= parent["ts"] < r["ts"] + r["dur"]
                       and (not r["name"].startswith("stage3::module[")
                            or r.get("tid") == parent.get("tid"))]
            if any(r["name"] == "stage3::optimizer_step" for r in matches):
                phase = "optimizer"
            elif any(r["name"] == "stage3::forward_backward" for r in matches):
                phase = "forward_backward"
            elif any(r["name"].startswith("stage3::iteration[") for r in matches):
                phase = "other_step"
            owners = [r for r in matches if r["name"].startswith("stage3::module[")]
            if owners:
                owner = min(owners, key=lambda r: r["dur"])["name"]
        key = (phase, owner, parent["name"] if parent else "", name)
        mapped[key][0] += 1
        mapped[key][1] += duration
        phase_intervals[phase].append((event["ts"], event["ts"] + duration))
    ordered = sorted(totals.items(), key=lambda item: -item[1][1])
    return {
        "file": path.name, "bytes": path.stat().st_size, "events": len(events),
        "categories": sorted({str(e.get("cat", "")) for e in events}),
        "kernel_count": len(kernels),
        "kernel_sum_us": sum(e["dur"] for e in kernels),
        "kernel_union_us": union_us([(e["ts"], e["ts"] + e["dur"]) for e in kernels]),
        "phase_union_us": {p: union_us(v) for p, v in phase_intervals.items()},
        "kernels": [{"name": n, "count": v[0], "duration_us": v[1]} for n, v in ordered],
        "attribution": [{"phase": k[0], "module": k[1], "cpu_op": k[2], "kernel": k[3],
                         "count": v[0], "duration_us": v[1]} for k, v in mapped.items()],
        "ranges": [{k: e.get(k) for k in ("name", "ts", "dur", "tid")} for e in ranges
                   if not e["name"].startswith("stage3::module[")],
        "examples": {"kernel": kernels[:3], "cpu": list(cpu.values())[:3]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/jovyan/hmoe-cloud/step-profile-20261013/profiles"))
    parser.add_argument("--trace-chunks", type=int, default=1)
    args = parser.parse_args()
    directories = sorted(p for p in args.root.iterdir() if p.is_dir() and p.name.startswith("500m_"))
    if len(directories) != 6:
        raise RuntimeError(f"Expected six configurations, found {len(directories)}")
    for directory in directories:
        summaries = sorted(directory.glob("chunk-*-summary.json"))
        traces = sorted(directory.glob("chunk-*.json.gz"))
        if len(summaries) != 10 or len(traces) != 10:
            raise RuntimeError(f"Incomplete profiles: {directory}")
        compact = []
        for path in summaries:
            payload = json.loads(path.read_text())
            groups = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
            for row in payload["rows"]:
                key = (row["name"], row["cpu_us"] > 0 or row["self_cpu_us"] > 0)
                value = groups[key]
                for index, field in enumerate(("count", "self_cpu_us", "cpu_us", "self_device_us", "device_us")):
                    value[index] += row[field]
            payload["rows"] = [dict(name=key[0], cpu_event=key[1], count=value[0],
                                    self_cpu_us=value[1], cpu_us=value[2],
                                    self_device_us=value[3], device_us=value[4])
                               for key, value in groups.items()]
            compact.append(payload)
        emit("compact_summaries", directory.name, compact)
        for path in traces[:args.trace_chunks]:
            print(f"STAGE3_READING {directory.name} {path.name}", flush=True)
            emit("trace", directory.name + "/" + path.name, trace_summary(path))
    print("STAGE3_READER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
