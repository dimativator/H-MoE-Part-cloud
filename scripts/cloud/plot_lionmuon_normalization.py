"""Plot measured W&B losses at their original optimizer steps."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


LABELS = {'original': 'Original LionMuon', 'rms': 'Exact RMS (α=0)',
          'neuron_half': 'Neuron-wise α=0.5', 'neuron_full': 'Neuron-wise α=1',
          'probability_down': 'Muon probability 0.35 → 0.05'}


def points(run: dict, metric: str) -> list[tuple[int, float]]:
    values = {}
    for row in [*run.get('history', []), run.get('summary', {})]:
        loss = row.get('final-val/loss', row.get(metric)) if metric == 'val/loss' else row.get(metric)
        if loss is not None and row.get('iter') is not None:
            values[int(row['iter'])] = float(loss)
    return sorted(values.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    value = json.loads(args.snapshot.read_text())
    runs = value['runs']
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    status = []
    for ax, metric, title in zip(axes, ('train/loss', 'val/loss'), ('Training loss', 'Validation loss')):
        for trial, label in LABELS.items():
            run = runs.get('lion-norm-500m-' + trial + '-20261117', {})
            curve = points(run, metric)
            if curve:
                ax.plot([x / 1000 for x, _ in curve], [y for _, y in curve], label=label, linewidth=1.4)
            if metric == 'val/loss':
                status.append(dict(trial=trial, state=run.get('state', 'not_started'),
                                   step=run.get('summary', {}).get('iter'), latest_validation=curve[-1] if curve else None))
        baseline = runs.get('7ys8z4vq', {})
        curve = points(baseline, metric)
        if curve:
            ax.plot([x / 1000 for x, _ in curve], [y for _, y in curve], '--', color='black',
                    label='Historical 7ys8z4vq (resume)', linewidth=1.5)
        if metric == 'val/loss' and baseline.get('summary', {}).get('final-val/loss') is not None:
            final = baseline['summary']['final-val/loss']
            ax.axhline(final, color='gray', linestyle=':', linewidth=1,
                       label=f'Historical final {final:.6f}')
        ax.set(title=title, xlabel='Optimizer step (thousands)', ylabel='Cross-entropy loss', xlim=(0, 75.457))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    timestamp = datetime.fromtimestamp(value['timestamp'], ZoneInfo('Europe/Moscow')).strftime('%d-%m-%Y %H:%M MSK')
    fig.suptitle('LionMuon · 500M · 1×C · 4 GPUs/run · global batch 128\n' + timestamp)
    fig.savefig(args.output / 'losses.png', dpi=170)
    fig.savefig(args.output / 'losses.pdf')
    (args.output / 'plot_summary.json').write_text(json.dumps(dict(timestamp=timestamp, trials=status,
        reference='https://wandb-radfan.ru/andrey/fp8-pretrain/runs/7ys8z4vq',
        limitation='Historical reference is a resumed decay segment; no earlier curve is fabricated. New runs are from scratch.'), indent=2))
    print(json.dumps(status))


if __name__ == '__main__':
    main()
