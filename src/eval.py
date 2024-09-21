import argparse
import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .api import train_axis_classification
from .api.eval import score
from .config import get_axis_classification_default_confg
from .utils.git import get_git_hash

warnings.simplefilter("ignore")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, nargs="+")
    parser.add_argument(
        "--accelerator", type=str, choices=["cpu", "gpu", "tpu"], default="gpu"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    submissions = []
    solutions = []
    for config in args.configs:
        target_dir = os.path.join(
            f"outputs/logs/{os.path.basename(config).replace('.yaml', '')}"
        )
        if os.path.exists(os.path.join(target_dir, "submission.csv")):
            logger.info("All cv")
            submissions.append(
                pd.read_csv(os.path.join(target_dir, "submission.csv"))
                .sort_values("row_id")
                .reset_index(drop=True)
            )
            solutions.append(pd.read_csv(os.path.join(target_dir, "solution.csv")))
        elif os.path.exists(os.path.join(target_dir, "submission_partial.csv")):
            logger.warning("Partial cv")
            submissions.append(
                pd.read_csv(os.path.join(target_dir, "submission_partial.csv"))
            )
            solutions.append(
                pd.read_csv(os.path.join(target_dir, "solution_partial.csv"))
            )
        else:
            raise FileNotFoundError()

    solution = solutions[0].copy()
    submission = submissions[0].copy()

    scr = score(solution=solution, submission=submission)
    logger.info(f"overall: {scr}")


if __name__ == "__main__":
    main()
