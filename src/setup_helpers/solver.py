from torch import nn, optim
from torch.optim import lr_scheduler

from ..optim import CosineAnnealingLRWarmup
from ..utils.yacs import CfgNode


def setup_optimizer(cfg: CfgNode, model: nn.Module) -> optim.Optimizer:
    optimizer_type = cfg.solver.optimizer
    lr = cfg.solver.base_lr
    weight_decay = cfg.solver.weight_decay
    momentum = cfg.solver.momentum
    if optimizer_type == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            # same in https://github.com/AtsunoriFujita/BirdCLEF-2023-Identify-bird-calls-in-soundscapes/blob/main/src/train_net.py
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=False,
        )
    else:
        raise ValueError()
    return optimizer


def setup_lr_scheduler(
    cfg: CfgNode, optimizer: optim.Optimizer
) -> lr_scheduler._LRScheduler:
    scheduler_type = cfg.solver.lr_scheduler
    if scheduler_type == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            cfg.solver.lr_scheduler_step,
            cfg.solver.lr_scheduler_gamma,
        )
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=cfg.solver.max_epochs, eta_min=1.0e-6
        )
    elif scheduler_type == "CosineAnnealingLRWarmup":
        scheduler = CosineAnnealingLRWarmup(
            optimizer=optimizer,
            warmup_epochs=cfg.solver.warmup_epochs,
            max_epochs=cfg.solver.max_epochs,
            warmup_start_lr=cfg.solver.warmup_start_lr,
            eta_min=1.0e-6,
        )
    else:
        raise ValueError()
    return scheduler
