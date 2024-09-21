import numpy as np
from albumentations.augmentations.geometric import \
    ShiftScaleRotate as AlbumentationsShiftScaleRotate

from ..data_container import DataContainer
from .transform_base import TransformBase


class ShiftScaleRotate(TransformBase):
    def __init__(
        self,
        shift_limit: float,
        scale_limit: float,
        rotate_limit: int,
        always_apply: bool,
        p: float,
    ) -> None:
        super().__init__(always_apply, p)

        self.shift_scale_rotate = AlbumentationsShiftScaleRotate(
            shift_limit, scale_limit, rotate_limit, always_apply=True, p=1.0
        )

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.img is not None

        data.img = self.shift_scale_rotate(image=data.img)["image"]
        return data
