"""Independent p2/p5 single-GPU LR grids and per-period four-GPU winners."""
import json
import math
import os
from pathlib import Path

STAGE = os.environ['SLIM_LION_STAGE']
if STAGE not in ('257m', '500m'):
    raise ValueError('Unknown SlimNorMuonLion stage')
PERIOD = int(os.environ['SLIM_LION_PERIOD'])
if PERIOD not in (2, 5):
    raise ValueError('Only authorized p2 and p5 are supported')
os.environ['MODE'] = 'matched'
os.environ['CLOUD_TRAIN_WORLD_SIZE'] = '1' if STAGE == '257m' else '4'
os.environ['CLOUD_DISABLE_CHECKPOINTS'] = '1'

from run_matched_normuon import campaign, preflight


def validate_final(record: dict, lr: str, period: int) -> None:
    expected = dict(optimizer='slim_normuon_lion', size=257, phase='1xC', lr=lr,
                    period=period, iter=39250, consumed_tokens=5144576000,
                    weight_decay=0.1, auxiliary_lr=0.001, auxiliary_weight_decay=0.0,
                    ns_steps=5, slim_update_rms=0.2, lion_lr_ratio=0.01,
                    matrix_momentum=0.99, direction_beta=0.95, matrix_beta2=0.95,
                    second_moment_clock='slim_steps_only', checkpoint_saving=False,
                    physical_world_size=1, compression_map='layer_map_llama_paper.json')
    if any(record.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Sweep final configuration does not match the authorized recipe')
    if not math.isfinite(record['final-val/loss']):
        raise RuntimeError('Nonfinite sweep final')


def select_winner(source: Path, period: int) -> dict:
    records = []
    for lr in campaign.LRS:
        path = source / f'lr{lr}' / 'finals' / f'slim_normuon_lion_257m_bf16_lr{lr}_wd1e-1_1xC_1gpu_p{period}.json'
        record = json.loads(path.read_text())
        validate_final(record, lr, period)
        records.append(record)
    return min(records, key=lambda r: r['final-val/loss'])


def main() -> None:
    preflight()
    if STAGE == '257m':
        lr = os.environ['LR']
        if lr not in campaign.LRS:
            raise ValueError('LR outside the authorized grid')
        size, target, wd = 257, 39250, '1e-1'
    else:
        winner = select_winner(Path(os.environ['SLIM_LION_SWEEP_ROOT']), PERIOD)
        lr = winner['lr']
        print('SLIM_LION_WINNER=' + json.dumps(winner), flush=True)
        size, target, wd = 500, 75457, '1e-4'
    print('SLIM_LION_CONFIG=' + json.dumps(dict(period=PERIOD, lr=lr, size=size,
          target=target, lion_lr_ratio=0.01, direction_beta=0.95, matrix_momentum=0.99,
          matrix_beta2=0.95, auxiliary_lr=0.001, checkpoint_saving=False)), flush=True)
    for phase in ('matched_smoke', '1xC'):
        campaign.train('slim_normuon_lion', lr, size, phase, target, weight_decay=wd)


if __name__ == '__main__':
    main()
