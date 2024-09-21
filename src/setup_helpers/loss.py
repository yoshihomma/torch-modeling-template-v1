import torch
from torch import nn

from ..utils.yacs import CfgNode


def setup_loss(cfg: CfgNode) -> nn.Module:
    loss_type = cfg.loss.type
    reduction = cfg.loss.reduction

    if loss_type == "CrossEntropyLoss":
        weight = torch.tensor(cfg.loss.weight, dtype=torch.float32)
        loss = nn.CrossEntropyLoss(reduction=reduction, weight=weight)
    elif loss_type == "MSELoss":
        loss = nn.MSELoss(reduction=reduction)
    elif loss_type == "BCEWithLogitsLoss":
        weight = torch.tensor(cfg.loss.weight, dtype=torch.float32)
        loss = nn.BCEWithLogitsLoss(reduction=reduction, weight=weight)
    elif loss_type == "BCELoss":
        weight = torch.tensor(cfg.loss.weight, dtype=torch.float32)
        loss = nn.BCELoss(reduction=reduction, weight=weight)
    elif loss_type == "NLLLoss":
        weight = torch.tensor(cfg.loss.weight, dtype=torch.float32)
        loss = nn.NLLLoss(reduction=reduction, weight=weight)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    return loss
