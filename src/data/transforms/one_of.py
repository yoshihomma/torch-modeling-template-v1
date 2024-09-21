import random
from typing import List

import numpy as np

from .transform_base import TransformBase


class OneOf(TransformBase):
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
    def __init__(self, always_apply: bool, p: float, transforms: List[TransformBase]):
        super().__init__(always_apply, p)
        self.transforms = transforms

        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def apply(self, x: np.ndarray):
        random_state = np.random.RandomState(random.randint(0, 2**32 - 1))
        t = random_state.choice(self.transforms, p=self.transforms_ps)
        x = t.apply(x)
        return x
