"""LionMuon with algebraic elementwise and spectral soft-sign maps.

The persistent EMA is updated once with beta2, after forming the beta1
direction. Both temperature distributions are fitted to that direction once
at the transition. DDP gradients and optimizer states are replicated.
"""

import math

import torch

from third_party.softsign.softmuon import zeropower_via_newtonschulz5


def fit_distribution(values: torch.Tensor) -> tuple[float, float]:
    """Median/MAD parameters of the folded-Cauchy magnitude model."""
    location = values.median()
    scale = (values - location).abs().median()
    return float(location), float(scale)


def magnitude_quantile(progress: float, stats: tuple[float, float]) -> float:
    """Invert the folded-Cauchy CDF by bracketing, including degenerate fits."""
    location, scale = stats
    location = abs(location)
    if progress <= 0:
        return 0.0
    if scale == 0:
        return location
    if not 0 < progress < 1:
        raise ValueError('Quantile progress must be in [0, 1)')

    def cdf(value: float) -> float:
        return (math.atan((value - location) / scale)
                + math.atan((value + location) / scale)) / math.pi

    lo = 0.0
    hi = location + scale * math.tan(math.pi * (1 + progress) / 4)
    hi = max(hi, location + scale)
    while cdf(hi) < progress:
        hi *= 2
    for _ in range(64):
        mid = (lo + hi) / 2
        if cdf(mid) < progress:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def temperature(stats: tuple[float, float], progress: float, epsilon: float,
                maximum: float = 1e9) -> float:
    q = magnitude_quantile(progress, stats)
    if q == 0:
        return math.inf
    y = 1 - epsilon
    return min(maximum, max(1.0, y / math.sqrt(1 - y * y) / q))


def elementwise_softsign(matrix: torch.Tensor, tau: float) -> torch.Tensor:
    if math.isinf(tau):
        return matrix.sign()
    return matrix / torch.hypot(matrix, torch.full_like(matrix, 1 / tau))


def spectral_softsign(matrix: torch.Tensor, tau: float, steps: int = 30) -> torch.Tensor:
    """Regularized polar map using convergent quintic Newton-Schulz.

The first columns of polar([H, I/tau]) equal
    (H H.T + I/tau**2)**(-1/2) H.
Iterating the augmented matrix avoids explicitly forming an ill-conditioned
inverse square root. Work on the smaller Gram matrix, in FP32 or FP64.
Unlike the fast hard-Muon coefficients, these coefficients converge to one.
"""
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        x = matrix.to(torch.float64 if matrix.dtype == torch.float64 else torch.float32)
        transposed = x.size(0) > x.size(1)
        if transposed:
            x = x.mT
        columns = x.size(1)
        identity = torch.eye(x.size(0), dtype=x.dtype, device=x.device)
        reg = 0.0 if math.isinf(tau) else 1 / tau
        augmented = torch.cat((x, reg * identity), dim=1)
        norm = augmented.norm().clamp_min(torch.finfo(x.dtype).tiny)
        augmented = augmented / norm
        for _ in range(steps):
            gram = augmented @ augmented.mT
            polynomial = -1.25 * gram + 0.375 * (gram @ gram)
            augmented = 1.875 * augmented + polynomial @ augmented
        out = augmented[:, :columns]
        return out.mT if transposed else out


class SoftLionMuon(torch.optim.Optimizer):
    """Use use_soft_lion_muon=True for matrices and False for auxiliary AdamW."""

    def __init__(self, params, *, total_steps: int, lr: float = 1e-3,
                 weight_decay: float = 0.1, beta1: float = 0.95, beta2: float = 0.99,
                 period: int = 5, lion_lr_ratio: float = 0.01,
                 start_fraction: float = 0.9, epsilon: float = 1e-4,
                 ns_steps: int = 6, soft_ns_steps: int = 30,
                 betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8):
        if not isinstance(total_steps, int) or total_steps < 2 or not 0 <= start_fraction < 1:
            raise ValueError('Require total_steps >= 2 and start_fraction in [0, 1)')
        if not all(0 <= b < 1 for b in (beta1, beta2, *betas)) or len(betas) != 2:
            raise ValueError('Require two AdamW betas and all betas in [0, 1)')
        if any(not isinstance(n, int) or n < 1 for n in (period, ns_steps, soft_ns_steps)):
            raise ValueError('Period and iteration counts must be positive integers')
        if not 0 < epsilon < 1 or not math.isfinite(lion_lr_ratio) or lion_lr_ratio <= 0:
            raise ValueError('Require epsilon in (0, 1) and positive finite Lion LR ratio')
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, beta1=beta1,
                                    beta2=beta2, lion_lr_ratio=lion_lr_ratio,
                                    use_soft_lion_muon=True, betas=betas, eps=eps))
        for group in self.param_groups:
            if any(not math.isfinite(group[k]) or group[k] < 0 for k in ('lr', 'weight_decay')):
                raise ValueError('Require finite nonnegative LR and weight decay')
            if not math.isfinite(group['eps']) or group['eps'] <= 0:
                raise ValueError('Require positive finite AdamW epsilon')
            for p in group['params']:
                if not p.is_floating_point() or (group['use_soft_lion_muon'] and p.ndim != 2):
                    raise ValueError('SoftLionMuon parameters must be floating matrices')
        self.control = dict(total_steps=total_steps, transition_start=int(total_steps * start_fraction),
                            period=period, epsilon=epsilon, ns_steps=ns_steps,
                            soft_ns_steps=soft_ns_steps)
        self.completed_steps = 0
        self.statistics = None
        self.last_temperatures = (math.inf, math.inf)
        self.last_is_muon = True

    def _direction(self, p: torch.Tensor, group: dict) -> torch.Tensor:
        dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
        grad = p.grad.to(dtype)
        state = self.state[p]
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(grad)
        return group['beta1'] * state['momentum_buffer'] + (1 - group['beta1']) * grad

    def _calibrate(self) -> None:
        entries, singular_values = [], []
        for group in self.param_groups:
            if not group['use_soft_lion_muon']:
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                direction = self._direction(p, group)
                entries.append(direction.flatten())
                singular_values.append(torch.linalg.svdvals(direction))
        if not entries:
            raise RuntimeError('No matrix gradients at temperature calibration')
        lion_stats = fit_distribution(torch.cat(entries))
        del entries
        muon_stats = fit_distribution(torch.cat(singular_values))
        self.statistics = (lion_stats, muon_stats)
        print('SOFT_LION_MUON_CALIBRATION=' + str(dict(step=self.completed_steps,
              lion_stats=lion_stats, muon_stats=muon_stats, source='pre_update_lion_direction')), flush=True)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.grad.is_sparse:
                    raise RuntimeError('SoftLionMuon does not support sparse gradients')
        step = self.completed_steps
        if step >= self.control['total_steps']:
            raise RuntimeError('SoftLionMuon exceeded its configured horizon')
        transition = self.control['transition_start']
        if step == transition:
            self._calibrate()
        taus = (math.inf, math.inf)
        if step >= transition:
            progress = (step - transition) / (self.control['total_steps'] - transition)
            taus = tuple(temperature(s, progress, self.control['epsilon']) for s in self.statistics)
        is_muon = step % self.control['period'] == 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
                grad = p.grad.to(dtype)
                state = self.state[p]
                eta = group['lr']
                if group['use_soft_lion_muon']:
                    direction = self._direction(p, group)
                    if is_muon:
                        if math.isinf(taus[1]):
                            update = zeropower_via_newtonschulz5(direction, self.control['ns_steps'])
                        else:
                            update = spectral_softsign(direction, taus[1], self.control['soft_ns_steps'])
                        update = update * (0.2 * math.sqrt(max(p.shape)))
                    else:
                        update = elementwise_softsign(direction, taus[0])
                        eta *= group['lion_lr_ratio']
                    state['momentum_buffer'].mul_(group['beta2']).add_(grad, alpha=1 - group['beta2'])
                else:
                    if 'step' not in state:
                        state.update(step=0, exp_avg=torch.zeros_like(grad), exp_avg_sq=torch.zeros_like(grad))
                    state['step'] += 1
                    b1, b2 = group['betas']
                    state['exp_avg'].lerp_(grad, 1 - b1)
                    state['exp_avg_sq'].lerp_(grad.square(), 1 - b2)
                    m = state['exp_avg'] / (1 - b1 ** state['step'])
                    v = state['exp_avg_sq'] / (1 - b2 ** state['step'])
                    update = m / (v.sqrt() + group['eps'])
                p.mul_(1 - eta * group['weight_decay'])
                p.add_(update.to(p.dtype), alpha=-eta)
        self.completed_steps += 1
        self.last_temperatures = taus
        self.last_is_muon = is_muon
        return loss

    def get_step_metrics(self) -> dict:
        return {'soft_lion_muon/is_muon': int(self.last_is_muon),
                'soft_lion_muon/soft_phase': int(self.statistics is not None),
                'soft_lion_muon/tau_lion': min(1e9, self.last_temperatures[0]),
                'soft_lion_muon/tau_muon': min(1e9, self.last_temperatures[1])}

    def state_dict(self) -> dict:
        result = super().state_dict()
        result['soft_lion_muon_control'] = dict(config=self.control, completed_steps=self.completed_steps,
              statistics=self.statistics, last_temperatures=self.last_temperatures,
              last_is_muon=self.last_is_muon)
        return result

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = dict(state_dict)
        control = state_dict.pop('soft_lion_muon_control')
        if control['config'] != self.control:
            raise ValueError('SoftLionMuon schedule differs from saved state')
        super().load_state_dict(state_dict)
        for saved, group in zip(state_dict['param_groups'], self.param_groups):
            for index, p in zip(saved['params'], group['params']):
                dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
                for key, value in state_dict['state'].get(index, {}).items():
                    if isinstance(value, torch.Tensor):
                        self.state[p][key] = value.to(device=p.device, dtype=dtype).clone()
        for key in ('completed_steps', 'statistics', 'last_temperatures', 'last_is_muon'):
            setattr(self, key, control[key])
