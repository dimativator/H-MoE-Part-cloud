#!/usr/bin/env python3
"""Verify exact saved final evaluation, no checkpoints, and original online W&B run."""
import argparse
import json
import math
from pathlib import Path
import time


def same_serialized_loss(remote: object, local: float) -> bool:
    # W&B JSON serialization can round the final binary float by a few ULPs.
    return (isinstance(remote, (int, float)) and not isinstance(remote, bool)
            and math.isfinite(remote)
            and math.isclose(remote, local, rel_tol=0.0, abs_tol=1e-12))


def final_loss(path: Path, target: int) -> float:
    events = set()
    losses = []
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item.get('iter') != target:
                continue
            events.add(item.get('event'))
            if item.get('event') == 'validation':
                loss = item.get('final-val/loss', item.get('val/loss'))
                if loss is not None:
                    losses.append(float(loss))
    required = {'validation', 'downstream', 'lm_eval'} if target == 75457 else {'validation'}
    if not required.issubset(events) or len(losses) != 1 or not math.isfinite(losses[0]):
        raise ValueError(f'Missing finite exact final evaluation at {target}: {path}')
    if any((path.parent / 'ckpts').rglob('*.pt')):
        raise ValueError(f'Unexpected checkpoint: {path.parent}')
    return losses[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--target', type=int, required=True)
    parser.add_argument('--opt', required=True)
    parser.add_argument('--wd', type=float, required=True)
    parser.add_argument('--lr', type=float, required=True)
    args = parser.parse_args()
    loss = final_loss(args.metrics, args.target)
    import wandb
    api = wandb.Api(timeout=30)
    for attempt in range(6):
        api.flush()
        run = api.run(f'andrey/fp8-pretrain/{args.run_id}')
        if same_serialized_loss(run.summary.get('final-val/loss'), loss):
            break
        if attempt == 5:
            raise ValueError(f'W&B final not uploaded: {args.run_id}')
        time.sleep(5)
    assert run.config['opt'] == args.opt
    assert run.config['weight_decay'] == args.wd and run.config['lr'] == args.lr
    assert run.config['dtype'] == 'bfloat16' and not run.config['fp8'] and not run.config['fp8_optim']
    assert run.config['iterations'] == args.target and run.config['no_local_save']
    assert run.config['latest_ckpt_interval'] == 0 and run.config['permanent_ckpt_interval'] == 0
    print('WD_FINAL_VERIFIED', json.dumps({'id': args.run_id, 'iter': args.target,
          'final_val_loss': loss, 'url': run.url}), flush=True)


if __name__ == '__main__':
    main()
