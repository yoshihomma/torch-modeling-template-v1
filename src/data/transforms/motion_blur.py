from albumentations.augmentations import MotionBlur as AlbumentationsMotionBlur

from ..data_container import DataContainer
from .transform_base import TransformBase


class MotionBlur(TransformBase):
    def __init__(
        self,
        blur_limit: int,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)

        self.motion_blur = AlbumentationsMotionBlur(
            blur_limit=blur_limit, always_apply=True, p=1.0
        )

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.img is not None

        x = data.img
        data.img = self.motion_blur(image=x)["image"]

        return data
