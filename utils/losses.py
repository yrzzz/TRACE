from typing import Optional
import torch
import torch.nn.functional as F

def ce_smooth(
    logits: torch.Tensor,
    y: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cross-entropy with optional label smoothing and class weights.
    Supports reduction='none' or 'mean'.
    """
    ls = float(label_smoothing)
    if ls <= 0.0:
        return F.cross_entropy(logits, y, weight=class_weights, reduction=reduction)

    # Prefer native implementation if available.
    try:
        return F.cross_entropy(
            logits,
            y,
            weight=class_weights,
            reduction=reduction,
            label_smoothing=ls,
        )
    except TypeError:
        pass

    # Fallback for older torch versions.
    if reduction not in ("none", "mean"):
        raise ValueError(f"Unsupported reduction: {reduction}")

    num_classes = logits.shape[1]
    if num_classes <= 1:
        raise ValueError("num_classes must be > 1 for label smoothing.")

    log_probs = F.log_softmax(logits, dim=1)
    one_hot = F.one_hot(y, num_classes=num_classes).to(log_probs.dtype)
    smooth = ls / float(num_classes - 1)
    target = one_hot * (1.0 - ls) + (1.0 - one_hot) * smooth
    loss = -(target * log_probs).sum(dim=1)
    if class_weights is not None:
        loss = loss * class_weights[y]

    if reduction == "none":
        return loss
    return loss.mean()
