import numpy as np
from albumentations.augmentations.geometric import \
    VerticalFlip as AlbumentationsVerticalFlip

from ..data_container import DataContainer
from .transform_base import TransformBase


class VerticalFlip(TransformBase):
    def __init__(self, always_apply: bool, p: float) -> None:
        super().__init__(always_apply, p)

        self.vertical_flip = AlbumentationsVerticalFlip(always_apply=True, p=1.0)

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.img is not None

        data.img = self.vertical_flip(image=data.img)["image"]
        return data
