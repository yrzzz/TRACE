import torch
import torch.nn.functional as F


def masked_pool(tokens_grid: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bsz, gh, gw, dim = tokens_grid.shape

    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask[:, 0, :, :]
    if mask.dim() != 3:
        raise ValueError(f"mask must have shape [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")

    if mask.shape[-2:] != (gh, gw):
        mask = F.interpolate(mask.unsqueeze(1).float(), size=(gh, gw), mode="nearest").squeeze(1)
    else:
        mask = mask.float()

    tokens_flat = tokens_grid.reshape(bsz, gh * gw, dim)
    mask_flat = mask.reshape(bsz, gh * gw)

    weighted = tokens_flat * mask_flat.unsqueeze(-1)
    denom = mask_flat.sum(dim=1, keepdim=True).clamp(min=eps)
    pooled = weighted.sum(dim=1) / denom
    return pooled
