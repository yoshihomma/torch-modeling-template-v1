from typing import Any, List, Optional, Tuple, Union

import pytorch_lightning as pl
from torch import Tensor, optim


class LightningModelBase(pl.LightningModule):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def configure_optimizers(
        self,
    ) -> Union[
        optim.Optimizer,
        Tuple[List[optim.Optimizer], List[optim.lr_scheduler._LRScheduler]],
    ]:
        super().configure_optimizers()
        if self.lr_scheduler is None:
            return self.optimizer
        return [self.optimizer], [self.lr_scheduler]

    @property
    def num_trainable_params(self) -> float:
        trainable_params = sum(
            p.numel() if not _is_lazy_weight_tensor(p) else 0
            for p in self.model.parameters()
            if p.requires_grad
        )
        return float(trainable_params)


def _is_lazy_weight_tensor(p: Tensor) -> bool:
    from torch.nn.parameter import UninitializedParameter

    if isinstance(p, UninitializedParameter):
        return True
    return False
