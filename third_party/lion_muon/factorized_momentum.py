"""LoRA-Pre simultaneous regression of LionMuon's normalized EMA history."""

import math

import torch


def initialize_factors(gradient: torch.Tensor, rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if gradient.ndim != 2 or not 0 < rank <= min(gradient.shape):
        raise ValueError("Factor rank must fit the matrix dimensions")
    generator = torch.Generator(device=gradient.device).manual_seed(seed)
    b = torch.zeros((gradient.shape[0], rank), device=gradient.device, dtype=torch.float32)
    a = torch.randn((rank, gradient.shape[1]), device=gradient.device,
                    dtype=torch.float32, generator=generator) * 0.02
    return b, a


@torch.no_grad()
def update_factors(b: torch.Tensor, a: torch.Tensor, gradient: torch.Tensor,
                   beta: float, damping: float) -> None:
    """Both right-hand sides use old factors. No dense history is retained."""
    if not 0 <= beta < 1 or damping <= 0:
        raise ValueError("Invalid factor decay or damping")
    gamma = 1 - math.sqrt(beta)
    g = gradient.float()
    aa = a @ a.T
    bb = b.T @ b
    aa.diagonal().add_(damping)
    bb.diagonal().add_(damping)
    b_target = torch.linalg.solve(aa, (g @ a.T).T).T
    a_target = torch.linalg.solve(bb, b.T @ g)
    b.lerp_(b_target, gamma)
    a.lerp_(a_target, gamma)
