# Adapted from zichongli5/NorMuon, commit c6989a8354730695d9f5a9faa6c55eeb24865209.
# MIT License, Copyright (c) 2025 zichongli5. See LICENSE in this directory.
import torch

# copied from https://github.com/KellerJordan/Muon/blob/master/muon.py
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X



def normuon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:  # for the case of conv filters
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = update.to(grad.dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)
    ################ NorMuon added ###################
    vnorm = update.norm(dim=(-2,-1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2,-1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10))) # This scaling keep the update norm the same as pre-normalization
    ##################################################
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


