import os
from glob import glob
from typing import Optional

import loguru
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from .. import setup_helpers
from ..utils.yacs import CfgNode


def train(
    cfg: CfgNode,
    data_dir: str,
    output_dir: str,
    ckpt_dir: Optional[str] = None,
    dir_name: Optional[str] = None,
    model_postfix: Optional[str] = None,
    accelerator="gpu",
    return_valid_pred=False,
    skip_exists=False,
    mode: str = "train",
    model_prefix: Optional[str] = None,
    save_model_policy: str = "best_valid",
) -> Optional[pd.DataFrame]:
    if dir_name is None:
        _dir_name = cfg.name
    else:
        _dir_name = dir_name

    if cfg.precision == 16:
        precision = "bf16"
    else:
        precision = 32
    print(f"Use precision: {precision}")

    model = setup_helpers.setup_model(cfg)

    train_transforms = setup_helpers.setup_transforms(cfg, "train")
    valid_transforms = setup_helpers.setup_transforms(cfg, "valid")

    train_data_loader, valid_data_loader = setup_helpers.seup(
        cfg=cfg,
        data_dir=data_dir,
        traintransforms=train_transforms,
        validtransforms=valid_transforms,
    )

    loss = setup_helpers.setup_loss(cfg)
    optimizer = setup_helpers.setup_optimizer(cfg, model)
    lr_scheduler = setup_helpers.setup_lr_scheduler(cfg, optimizer)

    ligthning_model = setup_helpers.setup_lightning_model(
        cfg=cfg, model=model, loss=loss, optimizer=optimizer, lr_scheduler=lr_scheduler
    )

    logger = TensorBoardLogger(os.path.join(output_dir, "logs"), _dir_name)

    if ckpt_dir is None:
        ckpt_dir = None
    else:
        ckpt_dir = os.path.join(ckpt_dir, _dir_name)
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="last-{epoch}",
            monitor="epoch",
            mode="max",
        ),
    ]

    if valid_data_loader is not None:
        if cfg.solver.validation_monitor == "valid_loss":
            mode = "min"
        else:
            raise ValueError(
                f"Invalid validation_monitor: {cfg.solver.validation_monitor}"
            )
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best_valid-{epoch}",
                monitor=cfg.solver.validation_monitor,
                mode=mode,
            )
        )

    if cfg.solver.early_stopping_patience > 0:
        loguru.logger.info(
            f"Early stopping enable: patience={cfg.solver.early_stopping_patience}"
        )
        if cfg.solver.validation_monitor == "valid_loss":
            mode = "min"
        else:
            raise ValueError(
                f"Invalid validation_monitor: {cfg.solver.validation_monitor}"
            )

        loguru.logger.info(
            f"Early stopping monitor: {cfg.solver.validation_monitor}, mode: {mode}"
        )
        callbacks.append(
            EarlyStopping(
                monitor=cfg.solver.validation_monitor,
                min_delta=0.00,
                patience=cfg.solver.early_stopping_patience,
                verbose=True,
                mode=mode,
            )
        )

    trainer = pl.Trainer(
        max_epochs=cfg.solver.max_epochs,
        devices=1,
        accelerator=accelerator,
        logger=logger,
        callbacks=callbacks,
        precision=precision,
        detect_anomaly=False,
    )

    # define save dirs
    if save_model_policy == "best_valid":
        assert valid_data_loader is not None
        target_checkpoint_expr = "best_valid-*.ckpt"
        model_name = "best_valid.pth"
    elif save_model_policy == "last_epoch":
        target_checkpoint_expr = "last-*.ckpt"
        model_name = "last_epoch.pth"
    else:
        raise ValueError(f"Invalid save_model_policy: {save_model_policy}")

    if model_prefix:
        model_name = model_prefix + model_name
    if model_postfix:
        model_name = os.path.splitext(model_name)[0] + model_postfix + ".pth"

    model_dir = os.path.join(output_dir, f"models/{cfg.name}")
    model_path = os.path.join(model_dir, model_name)

    if (not skip_exists) or (not os.path.exists(model_path)):
        trainer.fit(ligthning_model, train_data_loader, valid_data_loader)

        # save config file
        with open(
            os.path.join(
                output_dir,
                "logs",
                _dir_name,
                f"version_{trainer.logger.version}",
                "config.yaml",
            ),
            "w",
        ) as fp:
            yaml.safe_dump(cfg.convert_to_dict(), fp)

        # save model state-dict

        checkpoint_path = glob(
            os.path.join(
                output_dir,
                "logs",
                _dir_name,
                f"version_{trainer.logger.version}",
                f"checkpoints/{target_checkpoint_expr}",
            )
        )
        if len(checkpoint_path) != 1:
            raise ValueError()
        checkpoint_path = checkpoint_path[0]

        loguru.logger.info(f"Load checkpoint from: {checkpoint_path}")
        trained_ligthning_model = type(ligthning_model).load_from_checkpoint(
            checkpoint_path,
            model=model,
            loss=loss,
            optimizer=optimizer,
        )

        os.makedirs(os.path.abspath(os.path.join(model_path, os.pardir)), exist_ok=True)
        torch.save(trained_ligthning_model.model.state_dict(), model_path)
    else:
        loguru.logger.warning(f"Skip training: {cfg.dataset.cv_valid_fold_index}")

    if return_valid_pred:
        raise NotImplementedError()
        # assert valid_data_loader is not None
        # valid_dataset = valid_data_loader._dataset
        # pred_df = valid_dataset.df.copy()

        # model = setup_helpers.setup_classification_model(cfg)
        # model.load_state_dict(torch.load(model_path))

        # if accelerator == "gpu":
        #     model.to("cuda")
        # model.eval()

        # preds = []
        # with torch.no_grad():
        #     for batch in valid_data_loader:
        #         x = batch.img
        #         mask = batch.label_mask
        #         if accelerator == "gpu":
        #             x = x.to("cuda")
        #         pred = model.predict(x, mask).cpu().numpy()
        #         preds += list(pred)
        # return pred_df
