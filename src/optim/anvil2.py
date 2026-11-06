"""ANVIL II matrix updates with auxiliary Adam and cautious weight decay.

Reference: KellerJordan/modded-nanogpt, c924f68, train_gpt.py.
The slow stream engages halfway through this run. Auxiliary Adam follows the
active --opt muon path: shared LR, betas, epsilon, cadence, and decay coefficient,
with a sign gate added to decay. Matrix decay retains the
reference's extra LR factor. States are replicated FP32 tensors (DDP or one
device), without the reference trainer's weight mantissa or averaging buffers.
"""

import math

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT


ANVIL_MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)


def anvil_cascade(direction: Tensor) -> Tensor:
    """Six BF16 polynomial maps, computing the smaller Gram matrix."""
    x = direction.bfloat16()
    x = (x.float() / (1.05 * x.float().norm() + 1e-6)).bfloat16()
    tall = x.size(0) > x.size(1)
    for a, b, c in ANVIL_MAPS:
        gram = x.mT @ x if tall else x @ x.mT
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + (x @ polynomial if tall else polynomial @ x)
    return x.float()


class Anvil2(torch.optim.Optimizer):
    """ANVIL matrices selected by is_proj_params, auxiliary cautious AdamW."""

    def __init__(
        self, params: ParamsT, *, total_steps: int, lr: float = 0.02,
        weight_decay: float = 0.1, betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8, engage_fraction: float = 0.5,
        momentum_warmup_steps: int = 240, momentum_cooldown_steps: int = 50,
    ):
        if not isinstance(total_steps, int) or total_steps < 1:
            raise ValueError('ANVIL II requires a positive integer total_steps')
        if not math.isfinite(engage_fraction) or not 0 <= engage_fraction <= 1:
            raise ValueError('ANVIL II engage_fraction must be in [0, 1]')
        if any(not isinstance(n, int) or n < 0
               for n in (momentum_warmup_steps, momentum_cooldown_steps)):
            raise ValueError('ANVIL II momentum schedule lengths must be nonnegative integers')
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay,
                                    betas=betas, eps=eps, is_proj_params=True))
        for group in self.param_groups:
            if any(not math.isfinite(group[k]) or group[k] < 0 for k in ('lr', 'weight_decay')):
                raise ValueError('ANVIL II requires finite nonnegative LR and weight decay')
            if not math.isfinite(group['eps']) or group['eps'] <= 0:
                raise ValueError('ANVIL II requires positive finite Adam epsilon')
            if len(group['betas']) != 2 or any(not 0 <= b < 1 for b in group['betas']):
                raise ValueError('ANVIL II requires two Adam betas in [0, 1)')
            for p in group['params']:
                if isinstance(p, DTensor):
                    raise NotImplementedError('ANVIL II requires full parameters (one device or DDP)')
                if not p.is_floating_point() or (group['is_proj_params'] and p.ndim != 2):
                    raise ValueError('ANVIL II requires floating parameters and 2D hidden weights')
        self.control = dict(total_steps=total_steps,
                            engage_step=math.ceil(engage_fraction * total_steps),
                            momentum_warmup_steps=momentum_warmup_steps,
                            momentum_cooldown_steps=momentum_cooldown_steps)
        self.completed_steps = 0

    def _momentum(self) -> float:
        step = self.completed_steps
        warmup = self.control['momentum_warmup_steps']
        cooldown = self.control['momentum_cooldown_steps']
        if step < warmup:
            return 0.85 + 0.08 * step / warmup
        cooldown_start = self.control['total_steps'] - cooldown
        if cooldown and step > cooldown_start:
            return 0.93 - 0.08 * (step - cooldown_start) / cooldown
        return 0.93

    def _matrix_update(self, grad: Tensor, state: dict, mu: float, engaged: bool) -> Tensor:
        if not state:
            m, n = grad.shape
            state.update(fast_momentum=torch.zeros_like(grad),
                         slow_momentum=torch.zeros_like(grad),
                         lane_energy=torch.zeros((m, 1) if m >= n else (1, n),
                                                 dtype=torch.float32, device=grad.device))
        state['fast_momentum'].lerp_(grad, 1 - (0.85 if engaged else mu))
        state['slow_momentum'].lerp_(grad, 0.02)
        alpha = 0.4385 if engaged else 1.0
        momentum = alpha * state['fast_momentum'] + (1 - alpha) * state['slow_momentum']
        x = anvil_cascade((1 - mu) * grad + mu * momentum)
        axis = 1 if grad.size(0) >= grad.size(1) else 0
        state['lane_energy'].lerp_(x.square().mean(dim=axis, keepdim=True), 0.1)
        y = x * state['lane_energy'].clamp_min(1e-10).rsqrt()
        return y * (x.norm() / y.norm().clamp_min(1e-10))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.completed_steps >= self.control['total_steps']:
            raise RuntimeError('ANVIL II exceeded its configured training horizon')
        # Validate all gradients before mutating any weights or state.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.grad.layout != torch.strided:
                    raise RuntimeError('ANVIL II does not support sparse gradients')
        engaged = self.completed_steps >= self.control['engage_step']
        mu = self._momentum()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                with torch.autocast(device_type=p.device.type, enabled=False):
                    grad = p.grad.float()
                    state = self.state[p]
                    lr = group['lr']
                    if group['is_proj_params']:
                        update = self._matrix_update(grad, state, mu, engaged)
                        eta = lr * math.sqrt(max(1.0, p.size(0) / p.size(1)))
                        decay = eta * group['weight_decay'] * lr
                        gate = p.float() * state['slow_momentum'] >= 0
                    else:
                        if not state:
                            state.update(step=0, exp_avg=torch.zeros_like(grad),
                                         exp_avg_sq=torch.zeros_like(grad))
                        state['step'] += 1
                        b1, b2 = group['betas']
                        state['exp_avg'].lerp_(grad, 1 - b1)
                        state['exp_avg_sq'].lerp_(grad.square(), 1 - b2)
                        # Match MuonLite's Adam bias correction and epsilon placement.
                        update = state['exp_avg'] / (state['exp_avg_sq'].sqrt() + group['eps'])
                        scale = (1 - b1 ** state['step']) / math.sqrt(1 - b2 ** state['step'])
                        update = update / scale
                        eta = lr
                        decay = lr * group['weight_decay']
                        gate = p.float() * update > 0
                    # Compute the gate on old weights and round the combined update once.
                    weights = p.float()
                    p.copy_(weights - eta * update - decay * weights * gate)
        self.completed_steps += 1
        return loss

    def get_step_metrics(self) -> dict:
        return {'anvil2/engaged': int(self.completed_steps > self.control['engage_step']),
                'anvil2/completed_steps': self.completed_steps}

    def state_dict(self) -> dict:
        result = super().state_dict()
        result['anvil2_control'] = dict(config=self.control.copy(), completed_steps=self.completed_steps)
        return result

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = dict(state_dict)
        control = state_dict.pop('anvil2_control')
        if control['config'] != self.control:
            raise ValueError('ANVIL II schedule differs from saved state')
        completed = control['completed_steps']
        if not isinstance(completed, int) or not 0 <= completed <= self.control['total_steps']:
            raise ValueError('Invalid ANVIL II completed step count')
        super().load_state_dict(state_dict)
        # Restore from saved values directly: Optimizer otherwise rounds states to BF16.
        for saved, group in zip(state_dict['param_groups'], self.param_groups):
            for index, p in zip(saved['params'], group['params']):
                for key, value in state_dict['state'].get(index, {}).items():
                    if isinstance(value, Tensor):
                        self.state[p][key] = value.to(device=p.device, dtype=torch.float32).clone()
        self.completed_steps = completed


def build_anvil2(param_groups: list[dict], args, named_parameters: dict[str, Tensor]) -> Anvil2:
    """Match --opt muon grouping and Adam settings, adding cautious decay."""
    if getattr(args, 'fp8_optim', False) or getattr(args, 'optimizer_state_sharding', False):
        raise ValueError('ANVIL II requires replicated FP32 optimizer states')
    if getattr(args, 'non_proj_opt', 'adamw') != 'adamw':
        raise ValueError('ANVIL II auxiliary parameters require cautious AdamW')
    matrix_ids = {id(p) for name, p in named_parameters.items()
                  if p.ndim == 2 and not any(k in name for k in ('wte', 'lm_head', 'embed'))}
    matrices, auxiliary = [], []
    for group in param_groups:
        for p in group['params']:
            (matrices if id(p) in matrix_ids else auxiliary).append(p)
    # The active Muon path constructs its own groups from named parameters,
    # using the global LR and decay even for norms and model no-decay groups.
    groups = [dict(params=matrices, is_proj_params=True),
              dict(params=auxiliary, is_proj_params=False)]
    return Anvil2(groups, total_steps=args.iterations, lr=args.lr, weight_decay=args.weight_decay,
                  betas=(args.beta1, args.beta2), eps=1e-8,
                  engage_fraction=args.anvil2_engage_fraction,
                  momentum_warmup_steps=args.anvil2_momentum_warmup_steps,
                  momentum_cooldown_steps=args.anvil2_momentum_cooldown_steps)
