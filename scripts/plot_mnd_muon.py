"""Render validation curves from a downloaded MND_MUON_SNAPSHOT JSON object."""

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    snapshot = json.loads(args.snapshot.read_text())
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    curves = []
    for run, color in zip(snapshot['runs'], colors):
        points = []
        for row in run['validation']:
            loss = row.get('final-val/loss', row.get('val/loss'))
            if isinstance(loss, (int, float)) and math.isfinite(loss):
                points.append((row['consumed_tokens'] / 1e9, loss))
        if not points:
            continue
        points.sort()
        x, y = zip(*points)
        curves.append(points)
        label = f"LR {run['lr']:g} · {y[-1]:.4f}"
        for axis in axes:
            axis.plot(x, y, color=color, linewidth=1.8, label=label)
            axis.scatter(x[-1], y[-1], color=color, s=22)
    if not curves:
        plt.close(figure)
        raise RuntimeError('No finite validation measurements yet; refusing to draw an empty chart')
    frontier = max(points[-1][0] for points in curves)
    left = max(0, frontier * 0.6)
    axes[1].set_xlim(left, max(frontier * 1.02, left + 0.01))
    recent = [loss for points in curves for tokens, loss in points if tokens >= left]
    low, high = min(recent), max(recent)
    pad = max(0.01, (high - low) * 0.15)
    axes[1].set_ylim(low - pad, high + pad)
    for axis, title in zip(axes, ('Full training', 'Recent validation')):
        axis.set_title(title)
        axis.set_xlabel('Training tokens (B)')
        axis.set_ylabel('Validation loss (nats/token)')
        axis.grid(alpha=0.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(fontsize=9)
    figure.suptitle(f"MND-Muon · 257M · WD 0.1 · {snapshot['checked_utc']}", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    main()
