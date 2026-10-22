# Extracted from parshakova/matrix_normal_optimizers at 8a3b701f32a151363ae8b905605145cc48f81a3a.
# Apache-2.0. Copyright 2026 Matrix Normal Optimizers contributors.
# Modified: retain only MND-Muon kernels, unchanged from upstream.
import math

import torch


def inverse(diagonal, eps):
    return diagonal.clamp_min(eps).reciprocal()


def inverse_sqrt(diagonal, eps):
    return diagonal.clamp_min(eps).rsqrt()


def balance(row, col, eps):
    row_trace = row.sum() if row.ndim == 1 else row.trace()
    col_trace = col.sum() if col.ndim == 1 else col.trace()
    scale = (col_trace.clamp_min(eps) / row_trace.clamp_min(eps)).sqrt()
    return scale * row, col / scale


def variance_root(u, v, dimension, eps):
    # Rationalize the positive-v branch to avoid subtracting close numbers.
    u = u.clamp_min(0)
    discriminant = torch.hypot(v, 2 * math.sqrt(dimension) * u.sqrt())
    half_sum = 0.5 * discriminant + 0.5 * v.abs()
    root = torch.where(
        v >= 0,
        u / half_sum.clamp_min(torch.finfo(u.dtype).tiny),
        half_sum / dimension,
    )
    return root.square().clamp_min(eps)


_POLAR_COEFFICIENTS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
)


def polar_express(matrix, steps):
    work = matrix.to(torch.bfloat16)
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.T
    norm = work.norm()
    work = work / (torch.where(norm > 0, norm, torch.ones_like(norm)) * 1.01)
    for index in range(steps):
        position = min(index, len(_POLAR_COEFFICIENTS) - 1)
        a, b, c = _POLAR_COEFFICIENTS[position]
        if position < len(_POLAR_COEFFICIENTS) - 1:
            a, b, c = a / 1.01, b / 1.01**3, c / 1.01**5
        a, b, c = a / 1.01, b / 1.01, c / 1.01
        gram = work @ work.T
        polynomial = b * gram + (c * gram) @ gram
        work = a * work + polynomial @ work
    if transposed:
        work = work.T
    return work.to(matrix.dtype)


def mnd_muon_momentum(gradient, momentum, previous_direction, row, col, beta, eps):
    m, n = gradient.shape
    q = beta * torch.outer(row, col) + (1 - beta) * gradient.square()
    linear = (1 - beta) * gradient * previous_direction
    new_row = variance_root(q @ inverse(col, eps), linear @ inverse_sqrt(col, eps), n, eps)
    new_col = variance_root(
        q.T @ inverse(new_row, eps), linear.T @ inverse_sqrt(new_row, eps), m, eps
    )
    row, col = balance(new_row, new_col, eps)
    whitened = gradient * inverse_sqrt(row, eps)[:, None] * inverse_sqrt(col, eps)[None, :]
    momentum = beta * momentum + (1 - beta) * whitened
    return momentum, row, col
