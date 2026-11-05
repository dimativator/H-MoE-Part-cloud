"""Validate the complete eight-point 257M sweep before selecting a 500M LR."""

import json
import math
from pathlib import Path

SWEEPS = (
    ('/workspace-SR006.nfs3/dimativator/softmuon-aux-257m-20261105',
     ('1e-4', '5e-4', '1e-3', '2e-3')),
    ('/workspace-SR006.nfs3/dimativator/softmuon-aux-257m-high-lr-20261105',
     ('3e-3', '4e-3', '5e-3', '6e-3')),
)


def validate(record: dict, lr: str) -> None:
    expected = dict(optimizer='softmuon', algorithm='Algorithm3', lr=lr, size=257,
                    phase='1xC', iter=39250, consumed_tokens=5144576000,
                    weight_decay=0.1, auxiliary_weight_decay=0.1, auxiliary_lr=0.001,
                    physical_world_size=2, global_batch_size=128, checkpoint_saving=False,
                    seed=0, data_seed=1337, update_scale='original_shape_factor',
                    update_rms=None, nesterov=False, temperature_spectrum='updated_momentum',
                    sign_iters=35325, transition_iters=3925)
    if record.get('smoke') or any(record.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Unexpected sweep final configuration for LR {lr}')
    loss = record.get('final-val/loss')
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        raise ValueError(f'Nonfinite or missing final validation loss for LR {lr}')


def select(sweeps: tuple = SWEEPS) -> dict:
    records = []
    for root, lrs in sweeps:
        for lr in lrs:
            path = Path(root) / f'lr{lr}/finals/softmuon_257m_bf16_lr{lr}_wd1e-1_1xC_2gpu.json'
            record = json.loads(path.read_text())
            validate(record, lr)
            records.append(dict(matrix_lr=lr, final_validation_loss=record['final-val/loss'],
                                final_path=str(path)))
    if len(records) != 8 or len({r['matrix_lr'] for r in records}) != 8:
        raise ValueError('Exactly eight distinct, verified 257M finals required')
    winner = min(records, key=lambda r: (r['final_validation_loss'], float(r['matrix_lr'])))
    return dict(best_matrix_lr=winner['matrix_lr'], best_final_validation_loss=winner['final_validation_loss'],
                selection_metric='final-val/loss', selection_step=39250, trials=records)
