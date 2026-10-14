"""Paired moment codec and finite-horizon utility estimation for HorizonAdam.

The Gaussian rate-distortion spectrum guides a practical block quantizer. It
does not make scalar quantization, repeated compression, or training optimal.
"""

import math

import torch
from torch import Tensor


def pack_codes(codes: Tensor, bits: int) -> Tensor:
    """Pack each eight unsigned integer codes into exactly ``bits`` bytes."""
    shifts = torch.arange(8, device=codes.device, dtype=torch.int64) * bits
    words = (codes.to(torch.int64).reshape(*codes.shape[:-1], -1, 8) << shifts).sum(-1)
    byte_shifts = torch.arange(bits, device=codes.device, dtype=torch.int64) * 8
    return ((words.unsqueeze(-1) >> byte_shifts) & 255).to(torch.uint8).flatten(-2)


def unpack_codes(packed: Tensor, bits: int) -> Tensor:
    byte_shifts = torch.arange(bits, device=packed.device, dtype=torch.int64) * 8
    words = (packed.to(torch.int64).reshape(*packed.shape[:-1], -1, bits) << byte_shifts).sum(-1)
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64) * bits
    return ((words.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).flatten(-2).float()


def rounding_uniform(count: int, offset: int, seed: int, device: torch.device) -> Tensor:
    """Counter-based rounding noise, independent of rank and training RNG state."""
    values = (torch.arange(count, device=device, dtype=torch.int64) + offset + seed) & 0xFFFFFFFF
    values = ((values ^ (values >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    values = ((values ^ (values >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    values = values ^ (values >> 16)
    return (values >> 8).float() * (1.0 / (1 << 24))


def encode_blocks(
    values: Tensor, encoder: Tensor, bits: tuple[int, int],
    stochastic: bool, offset: int, seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode [blocks, coordinates, 2] values, storing local means and scales."""
    centers = values.mean(dim=1)
    latent = (values - centers.unsqueeze(1)) @ encoder.T
    scales = latent.abs().amax(dim=1).clamp_min_(1e-30)
    packed = []
    for mode, width in enumerate(bits):
        maximum = (1 << (width - 1)) - 1
        normalized = (latent[..., mode] / scales[:, mode, None] * maximum).clamp(-maximum, maximum)
        if stochastic:
            noise = rounding_uniform(normalized.numel(), offset, seed + mode * 0x80000000, values.device)
            lower = normalized.floor()
            codes = lower + (noise.view_as(normalized) < normalized - lower)
        else:
            codes = normalized.round()
        packed.append(pack_codes((codes + maximum).to(torch.int64), width))
    return torch.cat(packed, dim=-1), centers, scales


def decode_blocks(
    packed: Tensor, centers: Tensor, scales: Tensor, decoder: Tensor,
    bits: tuple[int, int], block_size: int,
) -> Tensor:
    split = block_size * bits[0] // 8
    latent = []
    for mode, codes in enumerate((packed[:, :split], packed[:, split:])):
        maximum = (1 << (bits[mode] - 1)) - 1
        latent.append((unpack_codes(codes, bits[mode]) - maximum) * (scales[:, mode, None] / maximum))
    return torch.stack(latent, dim=-1) @ decoder.T + centers.unsqueeze(1)


def fit_codec(source: Tensor, utility: Tensor, total_bits: int, ridge: float) -> dict:
    """Fit a 2x2 transform from probe state and its completed future window.

    Eigensystems are tiny and solved in CPU float64, then stored in FP32 on the
    parameter device. Whitening is regularized before the utility eigensolve.
    Tensor-wide probe covariance approximates the within-block residual law.
    """
    residual = source.detach().to(device="cpu", dtype=torch.float64)
    residual = residual - residual.mean(dim=0)
    scale = residual.square().mean(dim=0).sqrt().clamp_min_(1e-12)
    normalized = residual / scale
    covariance = normalized.T @ normalized / max(1, normalized.shape[0])
    covariance += ridge * torch.eye(2, dtype=torch.float64)
    metric = utility.detach().to(device="cpu", dtype=torch.float64) * scale[:, None] * scale[None, :]
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    root = (vectors * eigenvalues.sqrt()) @ vectors.T
    inverse_root = (vectors * eigenvalues.rsqrt()) @ vectors.T
    weighted = root @ metric @ root
    spectrum, modes = torch.linalg.eigh((weighted + weighted.T) * 0.5)
    spectrum, modes = spectrum.flip(0).clamp_min_(0), modes.flip(1)
    # Fix each eigenvector's sign so refreshes do not flip quantizer coordinates.
    for column in range(2):
        pivot = modes[:, column].abs().argmax()
        if modes[pivot, column] < 0:
            modes[:, column].neg_()
    candidates = [(first, total_bits - first) for first in range(2, 9)
                  if 2 <= total_bits - first <= 8]
    bits = min(candidates, key=lambda pair: (
        sum(float(spectrum[i]) * 2.0 ** (-2 * pair[i]) for i in range(2)),
        abs(pair[0] - pair[1]),
    ))
    encoder = (modes.T @ inverse_root) / scale[None, :]
    decoder = scale[:, None] * (root @ modes)
    return dict(encoder=encoder.to(source.device, torch.float32),
                decoder=decoder.to(source.device, torch.float32), bits=bits,
                spectrum=spectrum.to(source.device, torch.float32))


def start_probe_window(state: dict, floor: float) -> None:
    state["probe_source"] = torch.stack((state["probe_m"], state["probe_v"].clamp_min(floor).log()), -1)
    state["probe_dm"] = torch.ones_like(state["probe_m"])
    state["probe_dv"] = state["probe_v"].clone()
    state["probe_utility"] = torch.zeros(2, 2, device=state["probe_m"].device, dtype=torch.float32)
    state["window_steps"] = 0


def update_probes(state: dict, gradient: Tensor, group: dict) -> dict | None:
    """Use only observed gradients. Return a codec after a complete N-step window."""
    beta1, beta2 = group["betas"]
    floor = group["v_floor"]
    m, v = state["probe_m"], state["probe_v"]
    m.lerp_(gradient, 1 - beta1)
    v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    if "probe_source" not in state:
        v.clamp_min_(floor)
        start_probe_window(state, floor)
        return None

    state["probe_dm"].mul_(beta1)
    state["probe_dv"].mul_(beta2).mul_(v > floor)
    v.clamp_min_(floor)
    correction1 = 1 - beta1 ** state["step"]
    correction2 = 1 - beta2 ** state["step"]
    root = (v / correction2).sqrt()
    denominator = root + group["eps"]
    first = state["probe_dm"] / (correction1 * denominator)
    # Equivalent to the derivative with a cubic denominator, but these ratios
    # stay bounded when both variance and epsilon are very small.
    second = -0.5 * ((m / correction1) / denominator) * (state["probe_dv"] / v) * (root / denominator)
    jacobian = torch.stack((first, second), dim=-1) * group["lr"]
    state["probe_utility"].add_(jacobian.T @ jacobian / gradient.numel())
    state["window_steps"] += 1
    if state["window_steps"] < group["horizon"]:
        return None

    codec = fit_codec(state["probe_source"], state["probe_utility"], group["total_bits"], group["ridge"])
    start_probe_window(state, floor)
    return codec


def log_variance_bounds(floor: float) -> tuple[float, float]:
    return math.log(floor), math.log(torch.finfo(torch.float32).max) - 2
