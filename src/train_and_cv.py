import argparse
import os

import numpy as np
import pandas as pd
from loguru import logger

from .api import train
from .api.eval_axia_classification import (make_submission_and_solution_df,
                                           score)
from .config import get_axis_classification_default_confg
from .utils.git import get_git_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument(
        "--accelerator", type=str, choices=["cpu", "gpu", "tpu"], default="gpu"
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--output_dir", type=str, default="./outputs"
    )
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--num_partial_folds", type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    num_folds = min(
        get_axis_classification_default_confg().dataset.cv_num_folds,
        args.num_partial_folds,
    )
    assert num_folds <= get_axis_classification_default_confg().dataset.cv_num_folds

    valid_preds = []
    for i in range(num_folds):
        cfg = get_axis_classification_default_confg()
        cfg.merge_from_file(args.config)
        cfg.dataset.cv_valid_fold_index = i
        cfg.name = os.path.basename(args.config).split(".")[0]
        cfg.git_hash = get_git_hash()
        cfg.seed = i
        # overwrite pretrained model path
        if cfg.model.pretrained_model.startswith("dir:"):
            _model_dir = cfg.model.pretrained_model.replace("dir:", "")
            model_path = f"{_model_dir}/best_valid_fold{i}.pth"
            cfg.model.pretrained_model = f"file:{model_path}"
        if cfg.model.axial_backbone_pretrain.startswith("dir:"):
            _model_dir = cfg.model.axial_backbone_pretrain.replace("dir:", "")
            model_path = f"{_model_dir}/best_valid_fold{i}.pth"
            cfg.model.axial_backbone_pretrain = f"{model_path}"
        if cfg.model.sagittal_backbone_pretrain.startswith("dir:"):
            _model_dir = cfg.model.sagittal_backbone_pretrain.replace("dir:", "")
            model_path = f"{_model_dir}/best_valid_fold{i}.pth"
            cfg.model.sagittal_backbone_pretrain = f"{model_path}"
        cfg.freeze()

        dir_name = os.path.join(cfg.name, f"fold{i}")

  
        logger.info(f"Train Fold {i}")

        _valid_preds = train(
            cfg=cfg,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            ckpt_dir=args.ckpt_dir,
            dir_name=dir_name,
            accelerator=args.accelerator,
            return_valid_pred=True,
            model_postfix=f"_fold{i}",
            skip_exists=True,
        )
        valid_preds.append(_valid_preds)

    # TODO: implement eval

    # valid_preds = pd.concat(valid_preds, axis=0)
    # sub_df, solution_df = make_submission_and_solution_df(
    #     valid_preds, cfg.model.num_symptoms
    # )

    # if num_folds == cfg.dataset.cv_num_folds:
    #     sub_df.to_csv(
    #         os.path.join(args.output_dir, f"logs/{cfg.name}/submission.csv"),
    #         index=False,
    #     )
    #     solution_df.to_csv(
    #         os.path.join(args.output_dir, f"logs/{cfg.name}/solution.csv"),
    #         index=False,
    #     )
    # else:
    #     sub_df.to_csv(
    #         os.path.join(args.output_dir, f"logs/{cfg.name}/submission_partial.csv"),
    #         index=False,
    #     )
    #     solution_df.to_csv(
    #         os.path.join(args.output_dir, f"logs/{cfg.name}/solution_partial.csv"),
    #         index=False,
    #     )
    #     partial_file = os.path.join(args.output_dir, f"logs/{cfg.name}/partial.txt")
    #     with open(partial_file, "w") as f:
    #         f.write(f"Partial CV: {num_folds}/{cfg.dataset.cv_num_folds}\n")

    # logger.info(score(solution_df, sub_df))


if __name__ == "__main__":
    main()
