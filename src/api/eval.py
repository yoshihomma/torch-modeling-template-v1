from typing import Optional

from ..utils.yacs import CfgNode


# TODO: Implement score and eval functions

def score() -> float:
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
