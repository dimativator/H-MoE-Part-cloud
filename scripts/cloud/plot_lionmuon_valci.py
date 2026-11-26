"""Plot each measured validation population and the final sequence-bootstrap intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"p5": "LionMuon · every 5", "linear": "LionMuon · linear",
          "early": "LionMuon · favor early", "muon": "Muon"}


def plot(snapshot, output):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for role, color in zip(LABELS, plt.get_cmap("tab10").colors):
        run = snapshot["runs"].get(f"lion-valci-500m-{role}-20261126", {})
        summary = run.get("summary", {})
        for ax, key in zip(axes, ("val1024/loss", "val32768/loss")):
            rows = {r["iter"]:r[key] for r in run.get("history", []) if key in r and "iter" in r}
            if summary.get("iter") == 75457 and "final-val/loss" in summary and key in summary:
                rows[75457] = summary[key]
            steps = sorted(rows)
            ax.plot([i/1000 for i in steps], [rows[i] for i in steps],
                    marker="." if key == "val32768/loss" else None,
                    label=LABELS[role], color=color)
        if "val32768/low95" in summary and summary.get("iter") == 75457:
            value = summary["val32768/loss"]
            axes[2].errorbar(value, list(LABELS).index(role), fmt="o", color=color, capsize=4,
                             xerr=[[value-summary["val32768/low95"]],[summary["val32768/high95"]-value]])
    for ax, n in zip(axes, ("1,024", "32,768")):
        ax.set(title=f"{n} sequences", xlabel="Training step (thousands)", ylabel="Validation loss")
    axes[0].legend(fontsize=8)
    axes[2].set(title="Final loss · 95% bootstrap CI", xlabel="Validation loss",
                yticks=range(4), yticklabels=list(LABELS.values()), ylim=(3.5,-.5))
    for ax in axes:
        ax.grid(alpha=.2)
    output.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.snapshot.read_text()), args.output)
