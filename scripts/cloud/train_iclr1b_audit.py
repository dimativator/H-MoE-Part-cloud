"""Run the existing trainer with a read-only audit after its first optimizer step."""

from collections import Counter
import json
import os
from pathlib import Path
import runpy
import sys

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook


def audit(optimizer: torch.optim.Optimizer, _args: tuple, _kwargs: dict) -> None:
    parameters = {id(p): p for group in optimizer.param_groups for p in group['params']}
    count = sum(p.numel() for p in parameters.values())
    if count != int(os.environ['CLOUD_EXPECTED_PARAMETERS']):
        raise RuntimeError('Optimizer parameter coverage mismatch')
    moments = {'momentum', 'moment1', 'moment2', 'exp_avg', 'exp_avg_sq', 'mu', 'nu'}
    moment_elements = Counter()
    state_bytes = Counter()
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state_bytes[str(value.dtype)] += value.numel() * value.element_size()
                if key in moments:
                    moment_elements[str(value.dtype)] += value.numel()
    expected = 'torch.float8_e4m3fn' if os.environ['CLOUD_EXPECTED_STATE_FORMAT'] == 'fp8' else 'torch.float32'
    if not moment_elements or set(moment_elements) != {expected}:
        raise RuntimeError(f'Unexpected persisted optimizer moment types: {dict(moment_elements)}')
    record = dict(rank=int(os.environ['RANK']), optimizer=type(optimizer).__name__,
                  parameters=count, state_bytes=dict(state_bytes), moment_elements=dict(moment_elements),
                  expected_state_dtype=expected, verified=True)
    print('OPTIMIZER_STATE_AUDIT=' + json.dumps(record), flush=True)
    hook.remove()


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / 'src'), str(root)]
    hook = register_optimizer_step_post_hook(audit)
    sys.argv[0] = str(root / 'src/main.py')
    runpy.run_path(sys.argv[0], run_name='__main__')
