from albumentations.augmentations import GaussNoise as AlbumentationsGaussNoise

from ..data_container import DataContainer
from .transform_base import TransformBase


class GaussianNoise(TransformBase):
    def __init__(
        self,
        var_limit: tuple[float, float],
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)

        self.gaussian_noise = AlbumentationsGaussNoise(
            var_limit=var_limit, always_apply=True, p=1.0
        )

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.img is not None

        x = data.img
        data.img = self.gaussian_noise(image=x)["image"]

        return data
