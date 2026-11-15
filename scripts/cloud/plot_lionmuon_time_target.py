"""Render validation curves using measured training-only time per step."""
import argparse
import csv
import json
from pathlib import Path
import re


def read_events(text: str, marker: str) -> list[dict]:
    text = re.sub(r'\[\d+,\d+\]<(?:stdout|stderr)>:', '', text)
    decoder = json.JSONDecoder()
    values = []
    for match in re.finditer(re.escape(marker) + '=', text):
        values.append(decoder.raw_decode(text[match.end():].lstrip())[0])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timing-log', type=Path, required=True)
    parser.add_argument('--inspection-log', type=Path)
    parser.add_argument('--baseline-cache', type=Path, default=Path('/Users/dmitrii_andriianov/Documents/BRAIn/huawei_meff/report_iclr2027/bf16_500m'))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True, exist_ok=True)
    plans = read_events(args.timing_log.read_text(), 'TIMING_PLAN')
    if len(plans) != 1:
        raise RuntimeError('Exactly one verified timing plan required')
    plan = plans[0]
    (args.output / 'timing_plan.json').write_text(json.dumps(plan, indent=2))
    curves = read_events(args.inspection_log.read_text(), 'CURVE') if args.inspection_log else []
    latest = {row['name']: row for row in curves}
    def baseline(run: str) -> dict[int, float]:
        points = {}
        with (args.baseline_cache / (run + '.history.csv')).open() as stream:
            for row in csv.DictReader(stream):
                loss = row.get('final-val/loss') or row.get('val/loss')
                if loss and row.get('iter'):
                    points[int(float(row['iter']))] = float(loss)
        return points
    trunk = baseline('rzmnorvr')
    decay = baseline('6nksz43s')
    baselines = {'1xc': {**{i:l for i,l in trunk.items() if i <= 67911}, **decay}, '2xc': trunk}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    table = []
    for ax, budget, sheet_loss in zip(axes, ('1xc','2xc'), (2.761,2.699)):
        target = plan['targets'][budget]
        muon_time = plan['timing']['muon_reference']['seconds_per_step']
        lion_time = plan['timing']['lion_muon']['seconds_per_step']
        points = sorted((i,l) for i,l in baselines[budget].items() if 0 < i <= target['baseline_steps'])
        ax.plot([i*muon_time/3600 for i,l in points],[l for i,l in points],color='#335C81',label='Muon, historical')
        lion_points = sorted((r['iter'],r.get('final-val/loss',r.get('val/loss'))) for r in latest.get(budget,{}).get('validation',[]) if r['iter']>0)
        if lion_points:
            ax.plot([i*lion_time/3600 for i,l in lion_points],[l for i,l in lion_points],color='#D05A40',label='LionMuon p=5')
        ax.axhline(sheet_loss,color='#335C81',ls=':',lw=1,label=f'Sheet target {sheet_loss:.3f}')
        ax.axvline(target['lion_seconds']/3600,color='#D05A40',ls='--',lw=1,label='Planned LionMuon finish')
        ax.set(title=budget.upper().replace('XC','×C')+' reference budget',xlabel='Estimated training time (hours)',ylabel='Validation loss')
        ax.set_ylim(2.65,3.2); ax.grid(alpha=.18); ax.legend(fontsize=9)
        table.append(dict(budget=budget,table_target_loss=sheet_loss,**target))
    fig.suptitle('500M · BF16 · 4 GPUs · global batch 128',fontsize=15)
    fig.savefig(args.output/'loss_vs_training_time.png',dpi=180)
    fig.savefig(args.output/'loss_vs_training_time.pdf')
    (args.output/'comparison.json').write_text(json.dumps(dict(budgets=table,baseline_sources={
        'sheet':'https://docs.google.com/spreadsheets/d/1xpgQuKg_aAtzHIdOLUrXRK5cvyjpP_lpXOdpDCRaLe4/edit#gid=0',
        '1xc':'https://wandb-radfan.ru/andrey/fp8-pretrain/runs/6nksz43s',
        '2xc':'https://wandb-radfan.ru/andrey/fp8-pretrain/runs/rzmnorvr'},
        limitation='Muon 2xC cached history ends at step 150900 with loss 2.697199. Sheet target 2.699 is retained as a distinct horizontal target. Times exclude setup and evaluation. Baseline curves use a pre-existing W&B export.'),indent=2))
    print(args.output/'loss_vs_training_time.png')


if __name__ == '__main__':
    main()
