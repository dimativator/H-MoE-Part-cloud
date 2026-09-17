"""Pure-PyTorch optimizer variants used by the Stage 4 speed benchmark.

The implementations mirror the project baselines while keeping the Megatron
integration dependency-free: 2-D transformer weights use the memory-efficient
optimizer and embeddings, norms, and other parameters can be routed to AdamW
through MCore parameter overrides.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import torch


class _LowRankProjector:
    def __init__(
        self,
        rank: int,
        update_gap: int,
        *,
        random: bool,
        seed: int = 0,
    ) -> None:
        self.rank = rank
        self.update_gap = update_gap
        self.random = random
        self.seed = seed
        self.basis: torch.Tensor | None = None
        self.side: str | None = None

    def _refresh(self, grad: torch.Tensor) -> None:
        rows, columns = grad.shape
        rank = max(1, min(self.rank, rows, columns))
        self.side = "right" if rows >= columns else "left"
        if self.random:
            generator = torch.Generator(device=grad.device).manual_seed(self.seed)
            self.seed += 15
            if self.side == "right":
                shape = (rank, columns)
            else:
                shape = (rows, rank)
            self.basis = torch.randn(
                shape,
                generator=generator,
                device=grad.device,
                dtype=grad.dtype,
            ).div_(math.sqrt(rank))
            return

        matrix = grad.float()
        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        basis = vh[:rank] if self.side == "right" else u[:, :rank]
        self.basis = basis.to(dtype=grad.dtype)

    def project(self, grad: torch.Tensor, step: int) -> tuple[torch.Tensor, bool]:
        refreshed = self.basis is None or step % self.update_gap == 0
        if refreshed:
            self._refresh(grad)
        assert self.basis is not None and self.side is not None
        if self.side == "right":
            return grad @ self.basis.mT, refreshed
        return self.basis.mT @ grad, refreshed

    def project_back(self, low_rank: torch.Tensor) -> torch.Tensor:
        assert self.basis is not None and self.side is not None
        if self.side == "right":
            return low_rank @ self.basis
        return self.basis @ low_rank


class GaLoreAdamW(torch.optim.Optimizer):
    """Project-baseline GaLore AdamW with sign-SGD on the residual."""

    def __init__(
        self,
        params,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        rank: int,
        update_gap: int,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            update_gap=update_gap,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["projector"] = _LowRankProjector(
                        group["rank"], group["update_gap"], random=False
                    )
                projected, refreshed = state["projector"].project(
                    grad, state["step"]
                )
                if refreshed and "exp_avg" in state:
                    state["step"] = 0
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(projected)
                    state["exp_avg_sq"] = torch.zeros_like(projected)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(projected, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(projected, projected, value=1.0 - beta2)
                correction1 = 1.0 - beta1 ** state["step"]
                correction2 = 1.0 - beta2 ** state["step"]
                normalized = exp_avg.div(correction1).div_(
                    exp_avg_sq.div(correction2).sqrt().add_(group["eps"])
                )
                update = state["projector"].project_back(normalized)
                projected_grad = state["projector"].project_back(projected)
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
                parameter.add_(
                    grad.sub(projected_grad).sign(), alpha=-group["lr"]
                )
        return loss


class ApolloAdamW(torch.optim.Optimizer):
    """APOLLO with random projection and channel-wise gradient scaling."""

    def __init__(
        self,
        params,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        rank: int,
        update_gap: int,
        scale: float = 1.0,
        scale_type: str = "channel",
    ) -> None:
        if scale_type not in {"channel", "tensor"}:
            raise ValueError("scale_type must be 'channel' or 'tensor'")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            update_gap=update_gap,
            scale=scale,
            scale_type=scale_type,
        )
        super().__init__(params, defaults)
        seed = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                seed += 1
                self.state[parameter]["seed"] = seed

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                if "projector" not in state:
                    state["step"] = 0
                    state["projector"] = _LowRankProjector(
                        group["rank"],
                        group["update_gap"],
                        random=True,
                        seed=state["seed"],
                    )
                projected, _ = state["projector"].project(grad, state["step"])
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(projected)
                    state["exp_avg_sq"] = torch.zeros_like(projected)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(projected, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(projected, projected, value=1.0 - beta2)
                correction1 = 1.0 - beta1 ** state["step"]
                correction2 = 1.0 - beta2 ** state["step"]
                normalized = exp_avg.div(correction1).div_(
                    exp_avg_sq.div(correction2).sqrt().add_(group["eps"])
                )
                if group["scale_type"] == "tensor":
                    scaling = normalized.norm() / (projected.norm() + group["eps"])
                else:
                    norm_dim = 0 if normalized.shape[0] < normalized.shape[1] else 1
                    scaling = normalized.norm(dim=norm_dim) / (
                        projected.norm(dim=norm_dim) + group["eps"]
                    )
                    if norm_dim == 1:
                        scaling = scaling.unsqueeze(1)
                scaled_grad = grad * scaling * math.sqrt(group["scale"])
                if "scaled_grad_norm" in state:
                    scaled_norm = scaled_grad.norm()
                    limiter = torch.clamp(
                        scaled_norm / (state["scaled_grad_norm"] + group["eps"]),
                        min=1.01,
                    ) / 1.01
                    scaled_grad.div_(limiter)
                    state["scaled_grad_norm"] = scaled_norm / limiter
                else:
                    state["scaled_grad_norm"] = scaled_grad.norm()
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(scaled_grad, alpha=-group["lr"])
        return loss


class FrugalAdamW(torch.optim.Optimizer):
    """Coordinate FRUGAL AdamW with state on active columns only."""

    def __init__(
        self,
        params,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        density: float,
        update_gap: int,
        inactive_lr_scale: float = 1.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            density=density,
            update_gap=update_gap,
            inactive_lr_scale=inactive_lr_scale,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _new_indices(parameter: torch.Tensor, density: float) -> torch.Tensor:
        count = max(1, round(parameter.shape[1] * density))
        return torch.randperm(parameter.shape[1], device=parameter.device)[:count]

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["active_indices"] = self._new_indices(
                        parameter, group["density"]
                    )
                if state["step"] > 0 and state["step"] % group["update_gap"] == 0:
                    state["active_indices"] = self._new_indices(
                        parameter, group["density"]
                    )
                    state.pop("exp_avg", None)
                    state.pop("exp_avg_sq", None)
                    state["step"] = 0
                indices = state["active_indices"]
                active_grad = grad.index_select(1, indices)
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(active_grad)
                    state["exp_avg_sq"] = torch.zeros_like(active_grad)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(active_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    active_grad, active_grad, value=1.0 - beta2
                )
                correction1 = 1.0 - beta1 ** state["step"]
                correction2 = 1.0 - beta2 ** state["step"]
                active_update = exp_avg.div(correction1).div_(
                    exp_avg_sq.div(correction2).sqrt().add_(group["eps"])
                )
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                inactive_lr = group["lr"] * group["inactive_lr_scale"]
                parameter.add_(grad.sign(), alpha=-inactive_lr)
                correction = active_update.mul(-group["lr"]).add_(
                    active_grad.sign(), alpha=inactive_lr
                )
                parameter.index_add_(1, indices, correction)
        return loss


class SlimAdamW(torch.optim.Optimizer):
    """AdamW with a factored second moment selected per parameter group."""

    def __init__(
        self,
        params,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _squared(grad: torch.Tensor, dims: Sequence[int] | None) -> torch.Tensor:
        if dims is None:
            return grad.square()
        return grad.square().mean(dim=tuple(dims), keepdim=True)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            dims = group.get("slim_compress_dims")
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(
                        self._squared(parameter, dims)
                    )
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(
                    self._squared(grad, dims), alpha=1.0 - beta2
                )
                correction1 = 1.0 - beta1 ** state["step"]
                correction2 = 1.0 - beta2 ** state["step"]
                update = exp_avg.div(correction1).div_(
                    exp_avg_sq.div(correction2).sqrt().add_(group["eps"])
                )
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss


def no_op_state_init(optimizer, config=None) -> None:
    """The benchmark optimizers initialize shape-dependent state lazily."""

    del optimizer, config
