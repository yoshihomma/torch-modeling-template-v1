from typing import Optional

from torch import nn, optim

from .. import lightning_models as lm
from ..utils.yacs import CfgNode


# TODO: implement setup_lightning_model function
def setup_lightning_model(
    cfg: CfgNode,
    model: nn.Module,
    loss: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
) -> lm.LightningModel:
    # if cfg.lightning_model.type == "SomeLightningModel":
    #     return lm.SomeLightningModel(
    #         model=model, loss=loss, optimizer=optimizer, lr_scheduler=lr_scheduler
    #     )
    # else:
    #     raise ValueError()
    raise NotImplementedError()
