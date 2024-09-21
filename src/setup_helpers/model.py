import timm
import torch
from loguru import logger
from torch import nn

from ..utils.yacs import CfgNode


def _load_weights_skip_mismatch(model, weights_path):
    # Load Weights
    state_dict = torch.load(weights_path)
    model_dict = model.state_dict()

    # Iter models
    params = {}
    for (sdk, sfv), (mdk, mdv) in zip(state_dict.items(), model_dict.items()):
        if sfv.size() == mdv.size():
            params[sdk] = sfv
        else:
            logger.warning(
                "Skipping param: {}, {} != {}".format(sdk, sfv.size(), mdv.size())
            )

    # Reload + Skip
    model.load_state_dict(params, strict=False)
    logger.info(f"Loaded weights from: {weights_path}")
    return model


# TODO: Implement setup_model function
def setup_model(cfg: CfgNode) -> nn.Module:

    if cfg.model.pretrained_model.startswith("file:"):
        pretrained = False
    elif cfg.model.pretrained_model == "true":
        pretrained = True
    elif cfg.model.pretrained_model == "false":
        pretrained = False
    else:
        raise ValueError(
            f"Unexpected `model.pretrained_model` found: {cfg.model.pretrained_model}"
        )

    raise NotImplementedError()

    # model = ...
    # return model
