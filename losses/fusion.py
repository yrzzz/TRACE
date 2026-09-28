"""Label-free probability fusion, identical to the cooperative-loss mixture."""
import torch
import torch.nn.functional as F


def probability_mix(z_list, alpha, eps=1e-8):
    alpha_safe = alpha.clamp_min(eps)
    alpha_safe = alpha_safe / alpha_safe.sum(dim=1, keepdim=True).clamp_min(eps)
    p1, p2, p3 = [F.softmax(z, dim=1) for z in z_list]
    p_mix = (alpha_safe[:, 0:1] * p1 + alpha_safe[:, 1:2] * p2
             + alpha_safe[:, 2:3] * p3)
    p_mix = p_mix.clamp_min(eps)
    p_mix = p_mix / p_mix.sum(dim=1, keepdim=True).clamp_min(eps)
    return p_mix, torch.log(p_mix)
