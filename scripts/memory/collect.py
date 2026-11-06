"""Tabulate a scripts/memory/sweep.py output directory.

For every (compute precision, activation checkpointing, offload) setting it prints
  1. peak allocated memory (GB) per optimizer/state precision and microbatch,
  2. extra peak memory (GB) relative to a reference row (default: Muon, FP8 states),
     the layout of Tables 1 and 9 in the paper,
  3. the phase that sets the peak (F = forward/backward, O = optimizer step).
It also writes summary.csv with one row per case.

    python scripts/memory/collect.py results/mem500m [--metric peak_steady_gb]
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

NAME_RE = re.compile(
    r"^(?P<opt>[a-z_0-9]+)__(?P<states>fp32|fp8)-states__(?P<compute>bf16|fp8)"
    r"__mb(?P<mb>\d+)__ac-(?P<ac>none|mlp|full)__offload-(?P<offload>on|off)\.json$"
)
OPT_ORDER = ["adamw", "muon", "soap", "ademamix", "galore", "frugal", "frugal_mm", "apollo", "slimadam"]


def load(out_dir):
    cases = []
    for path in sorted(Path(out_dir).glob("*.json")):
        m = NAME_RE.match(path.name)
        if not m:
            continue
        data = json.loads(path.read_text())
        case = {**m.groupdict(), "mb": int(m["mb"]), "status": data.get("status", "ok")}
        case.update(data.get("summary", {}))
        cases.append(case)
    return cases


def fmt(value, signed=False):
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def table(rows, mbs, header):
    lines = [f"| {header} | " + " | ".join(str(mb) for mb in mbs) + " |",
             "|---|" + "---|" * len(mbs)]
    lines += [f"| {label} | " + " | ".join(cells) + " |" for label, cells in rows]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--metric", default="peak_gb",
                   help="peak_gb (all profiled steps, includes step 0) or peak_steady_gb")
    p.add_argument("--reference", default="muon:fp8", help="opt:states row to subtract")
    args = p.parse_args()

    cases = load(args.out_dir)
    if not cases:
        raise SystemExit(f"no sweep results in {args.out_dir}")
    ref_opt, ref_states = args.reference.split(":")

    groups = defaultdict(dict)
    for c in cases:
        groups[(c["compute"], c["ac"], c["offload"])][(c["opt"], c["states"], c["mb"])] = c

    report = []
    for (compute, ac, offload), cells in sorted(groups.items()):
        mbs = sorted({mb for _, _, mb in cells})
        keys = sorted({(o, s) for o, s, _ in cells},
                      key=lambda k: (OPT_ORDER.index(k[0]) if k[0] in OPT_ORDER else 99, k[1]))

        def value(opt, states, mb):
            c = cells.get((opt, states, mb))
            if c is None:
                return None
            return "OOM" if c["status"] == "oom" else c.get(args.metric)

        peak_rows, delta_rows, phase_rows = [], [], []
        for opt, states in keys:
            label = f"{opt} / {states} states"
            peaks = [value(opt, states, mb) for mb in mbs]
            peak_rows.append((label, [fmt(v) for v in peaks]))
            deltas = []
            for mb, v in zip(mbs, peaks):
                ref = value(ref_opt, ref_states, mb)
                ok = isinstance(v, float) and isinstance(ref, float)
                deltas.append(fmt(v - ref, signed=True) if ok else fmt(v if isinstance(v, str) else None))
            delta_rows.append((label, deltas))
            phases = []
            for mb in mbs:
                c = cells.get((opt, states, mb))
                phase = c.get("binding_phase_steady") if c and c["status"] == "ok" else None
                phases.append({"fwd_bwd": "F", "opt_step": "O"}.get(phase, "—"))
            phase_rows.append((label, phases))

        title = f"compute={compute}, activation checkpointing={ac}, offload={offload}"
        report += [
            f"## {title}", "",
            f"Peak allocated memory (GB, `{args.metric}`)", "", table(peak_rows, mbs, "microbatch"), "",
            f"Extra peak memory vs {ref_opt} / {ref_states} states (GB)", "",
            table(delta_rows, mbs, "microbatch"), "",
            "Phase that sets the peak (F = forward/backward, O = optimizer step)", "",
            table(phase_rows, mbs, "microbatch"), "",
        ]

    text = "\n".join(report)
    (args.out_dir / "summary.md").write_text(text)
    print(text)

    fields = ["opt", "states", "compute", "mb", "ac", "offload", "status", "peak_gb",
              "peak_steady_gb", "peak_fwd_bwd_gb", "peak_opt_step_gb", "binding_phase_steady",
              "allocated_start_gb", "saved_activations_fwd0_gb", "reserved_gb", "step_time_s_steady"]
    with open(args.out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases)


if __name__ == "__main__":
    main()
