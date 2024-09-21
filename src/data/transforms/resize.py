from typing import Optional

import numpy as np
from albumentations.augmentations.geometric import \
    Resize as AlbumentationsResize

from ..data_container import DataContainer
from .transform_base import TransformBase


class Resize(TransformBase):
    def __init__(self, width: int, height: int) -> None:
        super().__init__(always_apply=True, p=1.0)

        self.resize = AlbumentationsResize(height=height, width=width)

    def apply(self, data: DataContainer) -> DataContainer:
        data.img = self.resize(image=data.img)["image"]
        return data
