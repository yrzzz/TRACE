#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


def compute_coop_moe_loss(
    z_list: Sequence[torch.Tensor],
    alpha: torch.Tensor,
    y: torch.Tensor,
    ce_fn: Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor],
    temps: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    eps: float = 1e-8,
    lam_gate: float = 1.0,
    lam_exp: float = 0.5,
    lam_lb: float = 0.01,
    enable_lb: bool = True,
    enable_conflict_focus: bool = False,
    lambda_conflict: float = 0.5,
    conflict_metric: str = "js",
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Cooperative MoE objective with posterior responsibilities.

    Inputs:
      z_list = [z_cell, z_local, z_ctx], each [B, C]
      alpha  = gate_weights softmax(gate_logits), [B, 3]
      y      = labels, [B]

    Math implemented:
      p_k = softmax(z_k / T_k)
      p_mix = sum_k alpha_k * p_k
      z_mix = log(p_mix + eps)

      L_mix = CE_w_smooth(z_mix, y)

      r_k = stopgrad( alpha_k * p_k(y) / (sum_j alpha_j * p_j(y) + eps) )
      L_gate = E_i[ KL(r_i || alpha_i) ]
             = E_i[ sum_k r_ik * (log(r_ik+eps) - log(alpha_ik+eps)) ]

      L_experts = E_i[ sum_k r_ik * CE_w_smooth(z_k, y)_i ]

      Optional load-balance:
        alpha_bar = mean(alpha, dim=0)
        L_lb = sum_k (alpha_bar_k - 1/3)^2

      Optional conflict focus:
        w_i = 1 + lambda_conflict * conflict_i
      where conflict_i is JS(p1,p2,p3) or H(p_mix).
      w_i is applied to L_gate_i and L_experts_i only (not L_mix).
    """
    if len(z_list) != 3:
        raise ValueError(f"z_list must have 3 tensors, got {len(z_list)}")
    z1, z2, z3 = z_list
    if alpha.ndim != 2 or alpha.shape[1] != 3:
        raise ValueError(f"alpha must be [B,3], got shape={tuple(alpha.shape)}")
    if y.ndim != 1:
        raise ValueError(f"y must be [B], got shape={tuple(y.shape)}")
    if z1.shape != z2.shape or z1.shape != z3.shape:
        raise ValueError(f"expert logits must share shape, got {z1.shape}, {z2.shape}, {z3.shape}")
    if z1.shape[0] != alpha.shape[0] or z1.shape[0] != y.shape[0]:
        raise ValueError("batch size mismatch among z_list/alpha/y")

    e = float(eps)
    t1 = max(float(temps[0]), e)
    t2 = max(float(temps[1]), e)
    t3 = max(float(temps[2]), e)

    p1 = F.softmax(z1 / t1, dim=1)
    p2 = F.softmax(z2 / t2, dim=1)
    p3 = F.softmax(z3 / t3, dim=1)

    alpha_safe = alpha.clamp_min(e)
    alpha_safe = alpha_safe / alpha_safe.sum(dim=1, keepdim=True).clamp_min(e)

    # Probability-space fusion: p_mix = sum_k alpha_k * p_k
    p_mix = (
        alpha_safe[:, 0:1] * p1
        + alpha_safe[:, 1:2] * p2
        + alpha_safe[:, 2:3] * p3
    )
    p_mix = p_mix.clamp_min(e)
    p_mix = p_mix / p_mix.sum(dim=1, keepdim=True).clamp_min(e)
    z_mix = torch.log(p_mix)

    loss_mix = ce_fn(z_mix, y, "mean")

    idx = torch.arange(y.size(0), device=y.device)
    py1 = p1[idx, y]
    py2 = p2[idx, y]
    py3 = p3[idx, y]

    denom = alpha_safe[:, 0] * py1 + alpha_safe[:, 1] * py2 + alpha_safe[:, 2] * py3 + e
    r1 = (alpha_safe[:, 0] * py1) / denom
    r2 = (alpha_safe[:, 1] * py2) / denom
    r3 = (alpha_safe[:, 2] * py3) / denom
    r = torch.stack([r1, r2, r3], dim=1).detach()

    # L_gate_i = KL(r_i || alpha_i)
    l_gate_i = (r * (torch.log(r + e) - torch.log(alpha_safe + e))).sum(dim=1)

    ce1_i = ce_fn(z1, y, "none")
    ce2_i = ce_fn(z2, y, "none")
    ce3_i = ce_fn(z3, y, "none")
    l_exp_i = r[:, 0] * ce1_i + r[:, 1] * ce2_i + r[:, 2] * ce3_i

    if bool(enable_conflict_focus):
        metric = str(conflict_metric).lower()
        if metric not in ("js", "entropy"):
            raise ValueError(f"conflict_metric must be js|entropy, got {conflict_metric}")
        if metric == "js":
            m = (p1 + p2 + p3) / 3.0
            kl1 = (p1 * (torch.log(p1 + e) - torch.log(m + e))).sum(dim=1)
            kl2 = (p2 * (torch.log(p2 + e) - torch.log(m + e))).sum(dim=1)
            kl3 = (p3 * (torch.log(p3 + e) - torch.log(m + e))).sum(dim=1)
            conflict_i = (kl1 + kl2 + kl3) / 3.0
        else:
            conflict_i = -(p_mix * torch.log(p_mix + e)).sum(dim=1)
        w_i = 1.0 + float(lambda_conflict) * conflict_i
    else:
        conflict_i = torch.zeros_like(l_gate_i)
        w_i = torch.ones_like(l_gate_i)

    loss_gate = (w_i * l_gate_i).mean()
    loss_experts = (w_i * l_exp_i).mean()

    if bool(enable_lb):
        k = float(alpha_safe.shape[1])
        alpha_bar = alpha_safe.mean(dim=0)
        loss_lb = ((alpha_bar - (1.0 / k)) ** 2).sum()
    else:
        loss_lb = torch.zeros((), device=alpha_safe.device, dtype=alpha_safe.dtype)

    loss_total = (
        loss_mix
        + float(lam_gate) * loss_gate
        + float(lam_exp) * loss_experts
        + float(lam_lb) * loss_lb
    )

    alpha_entropy = -(alpha_safe * torch.log(alpha_safe + e)).sum(dim=1)
    stats = {
        "mix_loss": float(loss_mix.detach().item()),
        "gate_loss": float(loss_gate.detach().item()),
        "experts_loss": float(loss_experts.detach().item()),
        "lb_loss": float(loss_lb.detach().item()),
        "total_loss": float(loss_total.detach().item()),
        "mean_alpha1": float(alpha_safe[:, 0].mean().detach().item()),
        "mean_alpha2": float(alpha_safe[:, 1].mean().detach().item()),
        "mean_alpha3": float(alpha_safe[:, 2].mean().detach().item()),
        "alpha_entropy": float(alpha_entropy.mean().detach().item()),
        "mean_r1": float(r[:, 0].mean().detach().item()),
        "mean_r2": float(r[:, 1].mean().detach().item()),
        "mean_r3": float(r[:, 2].mean().detach().item()),
        "conflict_mean": float(conflict_i.mean().detach().item()),
        "weight_mean": float(w_i.mean().detach().item()),
    }
    details = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "alpha": alpha_safe,
        "p_mix": p_mix,
        "z_mix": z_mix,
        "r": r,
        "w_i": w_i,
        "conflict_i": conflict_i,
    }
    return loss_total, stats, details
