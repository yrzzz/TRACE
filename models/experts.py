import torch
import torch.nn as nn

def _build_mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
    if num_layers <= 0:
        return nn.Linear(in_dim, out_dim)

    layers = []
    dim = in_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class ExpertHeads(nn.Module):
    def __init__(
        self,
        uni_dim: int,
        dino_dim: int,
        proj_dim: int,
        num_classes: int,
        head_hidden_dim: int,
        head_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.proj_cell = nn.Sequential(nn.LayerNorm(uni_dim), nn.Linear(uni_dim, proj_dim))
        self.proj_local = nn.Sequential(nn.LayerNorm(uni_dim), nn.Linear(uni_dim, proj_dim))
        self.proj_ctx = nn.Sequential(nn.LayerNorm(dino_dim), nn.Linear(dino_dim, proj_dim))

        self.head_cell = _build_mlp(proj_dim, num_classes, head_hidden_dim, head_layers, dropout)
        self.head_local = _build_mlp(proj_dim, num_classes, head_hidden_dim, head_layers, dropout)
        self.head_ctx = _build_mlp(proj_dim, num_classes, head_hidden_dim, head_layers, dropout)

    def forward(self, emb_cell: torch.Tensor, emb_local: torch.Tensor, emb_ctx: torch.Tensor):
        proj_cell = self.proj_cell(emb_cell)
        proj_local = self.proj_local(emb_local)
        proj_ctx = self.proj_ctx(emb_ctx)

        z_cell = self.head_cell(proj_cell)
        z_local = self.head_local(proj_local)
        z_ctx = self.head_ctx(proj_ctx)

        return {
            "proj_cell": proj_cell,
            "proj_local": proj_local,
            "proj_ctx": proj_ctx,
            "logits_cell": z_cell,
            "logits_local": z_local,
            "logits_ctx": z_ctx,
        }
