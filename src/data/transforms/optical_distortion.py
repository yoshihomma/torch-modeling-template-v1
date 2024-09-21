from albumentations.augmentations import \
    OpticalDistortion as AlbumentationsOpticalDistortion

from ..data_container import DataContainer
from .transform_base import TransformBase


class OpticalDistortion(TransformBase):
    def __init__(
        self,
        distort_limit: float,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)

        self.optical_distortion = AlbumentationsOpticalDistortion(
            distort_limit=distort_limit, always_apply=True, p=1.0
        )

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.img is not None

        x = data.img
        data.img = self.optical_distortion(image=x)["image"]

        return data
