#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalTokenAttentionPool(nn.Module):
    """
    Single-query attention pooling over UNI patch tokens.
    """

    def __init__(
        self,
        token_dim: int,
        query_dim: int,
        tau: float = 1.0,
        cosine: bool = False,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.query_dim = int(query_dim)
        self.ln_tokens = nn.LayerNorm(self.token_dim)
        self.ln_query = nn.LayerNorm(self.query_dim)
        self.q_proj = nn.Linear(self.query_dim, self.token_dim)
        self.scale = math.sqrt(float(self.token_dim))
        self.tau = float(tau)
        self.cosine = bool(cosine)
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")

    def forward(
        self,
        tokens_grid: torch.Tensor,
        query_vec: torch.Tensor,
        logit_bias: torch.Tensor = None,
        bias_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens_grid: [B, gh, gw, D]
        query_vec:   [B, Dq]
        returns:
          pooled:    [B, D]
          attn_map:  [B, gh, gw]
        """
        bsz, gh, gw, dim = tokens_grid.shape
        if dim != self.token_dim:
            raise ValueError(f"Expected token dim={self.token_dim}, got {dim}")
        if query_vec.dim() != 2 or query_vec.shape[0] != bsz:
            raise ValueError(f"query_vec must be [B,Dq], got {tuple(query_vec.shape)}")

        tokens_flat = tokens_grid.reshape(bsz, gh * gw, dim)
        tokens_norm = self.ln_tokens(tokens_flat)
        query_norm = self.ln_query(query_vec)
        q = self.q_proj(query_norm)  # [B, D]

        if self.cosine:
            # Cosine attention: L2-normalize both sides, then temperature scaling.
            tokens_for_attn = F.normalize(tokens_norm, p=2, dim=-1)
            q_for_attn = F.normalize(q, p=2, dim=-1)
            logits = torch.bmm(tokens_for_attn, q_for_attn.unsqueeze(-1)).squeeze(-1)
        else:
            # Backward-compatible dot-product attention path.
            logits = torch.bmm(tokens_norm, q.unsqueeze(-1)).squeeze(-1) / self.scale

        if logit_bias is not None:
            if logit_bias.dim() == 3:
                logit_bias = logit_bias.reshape(bsz, gh * gw)
            if logit_bias.shape != logits.shape:
                raise ValueError(
                    f"logit_bias shape must be [B,gh*gw]={tuple(logits.shape)}, got {tuple(logit_bias.shape)}"
                )
            logits = logits + float(bias_scale) * logit_bias

        logits = logits / self.tau
        attn_flat = torch.softmax(logits, dim=1)
        pooled = torch.bmm(attn_flat.unsqueeze(1), tokens_flat).squeeze(1)  # [B, D]
        attn_map = attn_flat.reshape(bsz, gh, gw)
        return pooled, attn_map
