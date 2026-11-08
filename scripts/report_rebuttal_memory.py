"""Export logged measurements as CSV and paper-style LaTeX tables."""

import argparse
import csv
import json
from pathlib import Path
import statistics

NAMES = {"adam": "AdamW", "muon": "Muon", "soap": "SOAP", "ademamix": "AdEMAMix",
         "galore": "GaLore", "frugal": "FRUGAL", "frugal_muon_muon": "FRUGAL MM",
         "apollo": "APOLLO", "slim_adam": "SlimAdam"}
BATCHES = (1, 2, 4, 8, 16, 32)


def summarize(record: dict) -> list[dict]:
    trial = record["trial"]
    rows = []
    for rank in range(trial["dp"]):
        steps = [e for e in record["events"] if e["event"] == "step" and e["rank"] == rank]
        window = [e for e in steps if e["step"] <= 15]
        row = dict(**trial, rank=rank, status=record["status"], log=record["log"])
        if window:
            peak = max(window, key=lambda e: e["peak_allocated_bytes"])
            last = window[-1]
            row.update(peak_bytes=peak["peak_allocated_bytes"],
                       peak_gb=peak["peak_allocated_bytes"] / 1e9,
                       forward_backward_peak_gb=max(e["forward_backward_peak_bytes"] for e in window) / 1e9,
                       optimizer_peak_gb=max(e["optimizer_peak_bytes"] for e in window) / 1e9,
                       reserved_minus_allocated_gb=last["reserved_minus_allocated_bytes"] / 1e9,
                       median_step_ms=statistics.median(e["step_ms"] for e in window[5:] or window))
            row.update({name + "_gb": value["bytes"] / 1e9 for name, value in last["inventory"].items()})
        rows.append(row)
    return rows


def color(value: float) -> str:
    neutral = (251, 250, 250)
    target = (80, 140, 185) if value < 0 else (190, 105, 125)
    fraction = min(abs(value) / 5, 1)
    return "".join(f"{round(a + fraction * (b - a)):02X}" for a, b in zip(neutral, target))


def table(rows: list[dict], ac: str, offload: bool, regime: str) -> str:
    selected = {(r["method"], r["states"], r["mb"]): r for r in rows
                if r["suite"] == "b1" and r["ac"] == ac and r["offload"] == offload}
    lines = [r"\begin{table}[!htb]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l|rrrrrr}", r"\toprule",
             "Microbatch & " + " & ".join(map(str, BATCHES)) + r" \\", r"\midrule"]
    for method, label in NAMES.items():
        states = "fp8" if regime == "fp8" or method in ("adam", "muon", "soap", "ademamix") else "fp32"
        cells = []
        for batch in BATCHES:
            value = selected.get((method, states, batch))
            base = selected.get(("muon", "fp8", batch))
            if not value or not base:
                cells.append("NR")
            elif value["status"] != "success":
                cells.append(value["status"].upper())
            elif base["status"] != "success":
                cells.append("NR")
            else:
                delta = value["peak_gb"] - base["peak_gb"]
                cells.append(r"\colorbox[HTML]{" + color(delta) + "}{$" + f"{delta:+.2f}" + "$}")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    references = [selected.get(("muon", "fp8", b)) for b in BATCHES]
    lines += [r"\midrule", "Muon / FP8 (GB) & " + " & ".join(
        f"{r['peak_gb']:.2f}" if r and r["status"] == "success" else "NR" for r in references) + r" \\",
        r"\bottomrule", r"\end{tabular}"]
    state_text = ("All methods use FP8 states." if regime == "fp8" else
                  "AdamW, Muon, SOAP, and AdEMAMix use FP8 states. Structural methods use FP32 states.")
    lines += [r"\caption{Extra peak allocated memory (GB) relative to FP8 Muon in Megatron. "
              + f"The 500M model uses sequence length 1024, W16A16G32, {ac} activation checkpointing, "
              + ("and CPU activation offload. " if offload else "and no activation offload. ")
              + state_text + " Peaks cover the first 15 steps, including initialization and projection refreshes. "
              + "NR denotes an unavailable measurement.}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def loss_checks(records: list[dict]) -> list[dict]:
    lookup = {(r["trial"]["method"], r["trial"]["states"], r["trial"]["ac"], r["trial"]["offload"]): r
              for r in records if r["trial"]["suite"] == "b1" and r["trial"]["mb"] == 1}
    checks = []
    for key, record in lookup.items():
        method, states, ac, offload = key
        base = lookup.get((method, states, "none", False))
        if base is None:
            continue
        def losses(r):
            return {e["step"]: e["losses"] for e in r["events"]
                    if e["event"] == "step" and e["rank"] == 0}
        left, right = losses(base), losses(record)
        differences = [abs(left[s][name] - right[s][name]) for s in left.keys() & right.keys()
                       for name in left[s].keys() & right[s].keys()]
        checks.append(dict(method=method, states=states, ac=ac, offload=offload,
                           matched_steps=len(left.keys() & right.keys()),
                           max_abs_loss_difference=max(differences) if differences else None,
                           exact=bool(differences) and max(differences) == 0,
                           both_complete=base["status"] == record["status"] == "success"))
    return checks


def distributed_table(rows: list[dict], ac: str, mb: int) -> str:
    lines = [r"\begin{table}[!htb]", r"\centering\small",
             r"\begin{tabular}{l|rrrr}", r"\toprule",
             r"Data-parallel size & 1 & 2 & 4 & 8 \\", r"\midrule"]
    for method in ("adam", "muon", "frugal", "slim_adam"):
        cells = []
        for dp in (1, 2, 4, 8):
            selected = [r for r in rows if r["suite"] == "b4" and r["method"] == method
                        and r["dp"] == dp and r["mb"] == mb and r["ac"] == ac]
            cells.append(f"{max(r['peak_gb'] for r in selected):.2f}"
                         if len(selected) == dp and all(r["status"] == "success" for r in selected)
                         else "NR")
        lines.append(NAMES[method] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Maximum per-rank peak allocated memory (GB) under whole-parameter "
              + "optimizer-state sharding. Model weights, gradients, and FP32 master weights remain replicated. "
              + f"The per-rank microbatch is {mb} with {ac} activation checkpointing. "
              + "AdamW and Muon use FP8 states. FRUGAL and SlimAdam use FP32 states. "
              + "The global batch equals the per-rank microbatch times the data-parallel size.}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def breakdown_table(rows: list[dict], mb: int) -> str:
    lines = [r"\begin{table}[!htb]", r"\centering\scriptsize",
             r"\begin{tabular}{l|rrrrrrr}", r"\toprule",
             r"Optimizer & Model & Master & Grads & State & Fwd/bwd peak & Opt. peak & Peak \\", r"\midrule"]
    for r in rows:
        if r["suite"] != "b2" or r["mb"] != mb:
            continue
        name = NAMES[r["method"]] + (" (fused fallback)" if r["fused"] else "")
        fields = ("model_gb", "master_gb", "grads_gb", "optimizer_state_gb",
                  "forward_backward_peak_gb", "optimizer_peak_gb", "peak_gb")
        cells = [f"{r[k]:.2f}" if k in r and r["status"] == "success" else "NR" for k in fields]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Memory components and phase peaks (GB) for the 500M Megatron model at microbatch "
              + str(mb) + ". Persistent storage is measured after step 15. Phase peaks include all live allocations "
              + "and must not be added together. Allocator traces provide allocation-site diagnostics.}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in args.input.glob("*/result.json")]
    rows = [row for record in records for row in summarize(record)]
    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (args.output / "per_rank_memory.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "loss_checks.json").write_text(json.dumps(loss_checks(records), indent=2))
    for ac in ("none", "selective", "full"):
        for offload in (False, True):
            for regime in ("mixed", "fp8"):
                name = f"b1_{ac}_offload{int(offload)}_{regime}.tex"
                (args.output / name).write_text(table(rows, ac, offload, regime))
    for mb in (1, 32):
        (args.output / f"b2_breakdown_mb{mb}.tex").write_text(breakdown_table(rows, mb))
        for ac in ("none", "full"):
            (args.output / f"b4_{ac}_mb{mb}.tex").write_text(distributed_table(rows, ac, mb))
    (args.output / "coverage.json").write_text(json.dumps(dict(
        trials=len(records), successful=sum(r["status"] == "success" for r in records),
        oom=sum(r["status"] == "oom" for r in records),
        failed=sum(r["status"] == "failed" for r in records)), indent=2))


if __name__ == "__main__":
    main()
