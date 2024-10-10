"""This file defines various uncertainty functions that can be used to adjust the conformal thresholds"""
import torch


def logit_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of prediction distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def smx_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of prediction distribution from softmax scores."""
    return -(x * x.log()).sum(1)
