from albumentations.augmentations import \
    RandomBrightnessContrast as AlbumentationsRandomBrightnessContrast

from ..data_container import DataContainer
from .transform_base import TransformBase


class RandomBrightnessContrast(TransformBase):
    def __init__(
        self,
        brightness_limit: float,
        contrast_limit: float,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)

        self.random_brightness_contrast = AlbumentationsRandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            always_apply=True,
            p=1.0,
        )

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.img is not None

        x = data.img
        data.img = self.random_brightness_contrast(image=x)["image"]

        return data
