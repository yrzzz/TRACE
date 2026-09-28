from contextlib import nullcontext
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .uni_tokens import uni_forward_tokens
from .dino_tokens import dino_forward_tokens
from .pooling import masked_pool
from .experts import ExpertHeads
from .gate import Gate, uncertainty_features
from .local_attn_pool import LocalTokenAttentionPool
from losses.fusion import probability_mix

class CellExpertModelE1AttnE2E3Attn(nn.Module):
    """
    New 3-expert model:
      - E1: UNI masked pooling
      - E2: UNI local attention pooling
      - E3: DINO context attention pooling
    """

    def __init__(
        self,
        num_classes: int,
        uni_dim: int,
        dino_dim: int,
        proj_dim: int = 512,
        head_hidden_dim: int = 256,
        head_layers: int = 1,
        gate_hidden_dim: int = 256,
        gate_layers: int = 2,
        dropout: float = 0.1,
        uni_model: nn.Module = None,
        dino_model: nn.Module = None,
        e2_attn_tau: float = 2.0,
        e3_attn_tau: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.uni = uni_model
        self.dino = dino_model
        self.freeze_backbones = True
        self.eps = float(eps)

        self.experts = ExpertHeads(
            uni_dim=uni_dim,
            dino_dim=dino_dim,
            proj_dim=proj_dim,
            num_classes=num_classes,
            head_hidden_dim=head_hidden_dim,
            head_layers=head_layers,
            dropout=dropout,
        )

        self.local_attn_pool = LocalTokenAttentionPool(
            token_dim=uni_dim,
            query_dim=uni_dim,
            tau=float(e2_attn_tau),
            cosine=True,
        )
        self.ctx_attn_pool = LocalTokenAttentionPool(
            token_dim=dino_dim,
            query_dim=uni_dim,
            tau=float(e3_attn_tau),
            cosine=True,
        )

        gate_input_dim = proj_dim * 3 + 9
        self.gate = Gate(
            input_dim=gate_input_dim,
            hidden_dim=gate_hidden_dim,
            num_layers=gate_layers,
            dropout=dropout,
        )

        if self.freeze_backbones:
            if self.uni is not None:
                for p in self.uni.parameters():
                    p.requires_grad = False
                self.uni.eval()
            if self.dino is not None:
                for p in self.dino.parameters():
                    p.requires_grad = False
                self.dino.eval()

    def forward(
        self,
        cell_img_224: torch.Tensor,
        cell_mask_224: torch.Tensor,
        ctx_img: torch.Tensor,
        cell_pos_pxpy: torch.Tensor,
        return_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if self.uni is None or self.dino is None:
            raise ValueError("Both UNI and DINO backbones are required.")

        context = torch.no_grad() if self.freeze_backbones else nullcontext()
        with context:
            uni_tokens = uni_forward_tokens(self.uni, cell_img_224)
            emb_cell = masked_pool(uni_tokens, cell_mask_224)

            dino_tokens, patch_size = dino_forward_tokens(self.dino, ctx_img)

        emb_local, local_attn_map = self.local_attn_pool(uni_tokens, emb_cell)
        emb_ctx, ctx_attn_map = self.ctx_attn_pool(dino_tokens, emb_cell)

        expert_out = self.experts(emb_cell, emb_local, emb_ctx)
        z_cell = expert_out["logits_cell"]
        z_local = expert_out["logits_local"]
        z_ctx = expert_out["logits_ctx"]

        unc_cell = uncertainty_features(z_cell)
        unc_local = uncertainty_features(z_local)
        unc_ctx = uncertainty_features(z_ctx)

        gate_input = torch.cat(
            [
                expert_out["proj_cell"],
                expert_out["proj_local"],
                expert_out["proj_ctx"],
                unc_cell,
                unc_local,
                unc_ctx,
            ],
            dim=1,
        )
        gate_logits = self.gate(gate_input)
        gate_weights = F.softmax(gate_logits, dim=1)

        probs, fused = probability_mix((z_cell, z_local, z_ctx), gate_weights, eps=self.eps)

        out = {
            "logits": fused,
            "probs": probs,
            "logits_cell": z_cell,
            "logits_local": z_local,
            "logits_ctx": z_ctx,
            "gate_weights": gate_weights,
            "emb_cell": emb_cell,
            "emb_local": emb_local,
            "emb_ctx": emb_ctx,
        }
        if return_attn:
            out["local_attn_map"] = local_attn_map
            out["ctx_attn_map"] = ctx_attn_map
        return out
