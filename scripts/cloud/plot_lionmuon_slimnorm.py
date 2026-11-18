"""Validation curves and a measured recent-step zoom, with matched references."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from plot_lionmuon_normalization import points

TRIALS = ('neuron_no_rms', 'neuron_rms_down', 'neuron_no_rms_down',
          'slim_no_rms', 'slim_rms_down', 'slim_no_rms_down')
LABELS = ('Row: no RMS, periodic', 'Row: RMS, p↓', 'Row: no RMS, p↓',
          'Slim: no RMS, periodic', 'Slim: RMS, p↓', 'Slim: no RMS, p↓')
REFERENCES = {'lion-norm-500m-original-20261117': 'Original LionMuon (previous)',
              'lion-norm-500m-probability_down-20261117': 'p 0.35→0.05 (previous)'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    value = json.loads(args.snapshot.read_text())
    reference = json.loads(args.references.read_text())
    curves = []
    status = []
    for trial, label in zip(TRIALS, LABELS):
        run = value['runs'].get(f'lion-slimnorm-500m-{trial}-20261118', {})
        curve = [(x, y) for x, y in points(run, 'val/loss') if math.isfinite(y)]
        curves.append((label, curve, False))
        status.append(dict(trial=trial, state=run.get('state', 'not_started'),
                           step=run.get('summary', {}).get('iter'), latest_validation=curve[-1] if curve else None))
    latest = max((x for _, c, _ in curves for x, _ in c), default=0)
    for run_id, label in REFERENCES.items():
        curves.append((label, points(reference['runs'][run_id], 'val/loss'), True))
    # Zoom uses the recent portion actually reached by new runs, never future reference losses.
    right = max(latest, 1000)
    left = max(500, right - 10000)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    colors = list(plt.get_cmap('tab10').colors[:6]) + ['black', '#7b3294']
    for ax in axes:
        for (label, curve, is_reference), color in zip(curves, colors):
            if curve:
                ax.plot([x/1000 for x, _ in curve], [y for _, y in curve],
                        label=label, color=color, linestyle='--' if is_reference else '-', linewidth=1.5)
        ax.grid(alpha=.2)
        ax.set(xlabel='Optimizer steps (thousands)', ylabel='Validation cross-entropy')
    axes[0].set(title='Measured validation curves', xlim=(0, 75.457))
    axes[1].set(title='Recent validation zoom', xlim=(left/1000, right/1000))
    ys = [y for _, curve, _ in curves for x, y in curve if left <= x <= right and math.isfinite(y)]
    if ys:
        pad = max(.002, (max(ys)-min(ys))*.1)
        axes[1].set_ylim(min(ys)-pad, max(ys)+pad)
    if not latest:
        axes[1].text(.5, .9, 'New runs have no measured validation yet', ha='center', transform=axes[1].transAxes)
    axes[0].legend(fontsize=8)
    timestamp = datetime.fromtimestamp(value['timestamp'], timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    fig.suptitle('LionMuon α=1 · 500M / 1×C · 4 GPUs/run · seed 0\n' + timestamp)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'validation_zoom.png', dpi=180)
    fig.savefig(args.output/'validation_zoom.pdf')
    (args.output/'plot_summary.json').write_text(json.dumps(dict(timestamp=timestamp, runs=status), indent=2))
    print(json.dumps(status))


if __name__ == '__main__':
    main()
