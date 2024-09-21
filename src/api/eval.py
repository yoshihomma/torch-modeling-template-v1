from typing import Optional
import pandas as pd

from ..utils.yacs import CfgNode


# TODO: Implement score and eval functions

def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    raise NotImplementedError()


def eval(
    cfg: CfgNode,
    data_dir: str,
    output_dir: str,
    cv_hash: Optional[str],
) -> float:

    raise NotImplementedError()

    cv_score = score()
    return cv_score
