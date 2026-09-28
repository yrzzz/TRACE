#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


def uncertainty_features(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    eps = 1e-8
    entropy = -(probs * (probs + eps).log()).sum(dim=1)

    k = min(2, probs.shape[1])
    topk = torch.topk(probs, k=k, dim=1).values
    maxprob = topk[:, 0]
    if k == 2:
        margin = topk[:, 0] - topk[:, 1]
    else:
        margin = topk[:, 0]

    return torch.stack([entropy, margin, maxprob], dim=1)


class Gate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_experts: int = 3,
    ) -> None:
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
