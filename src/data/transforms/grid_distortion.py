import numpy as np
from albumentations.augmentations.geometric import \
    GridDistortion as AlbumentationsGridDistortion

from ..data_container import DataContainer
from .transform_base import TransformBase


class GridDistortion(TransformBase):
    def __init__(
        self,
        num_steps: int,
        distort_limit: float,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)

        self.grid_distortion = AlbumentationsGridDistortion(
            num_steps=num_steps, distort_limit=distort_limit, always_apply=True, p=1.0
        )

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.img is not None

        x = data.img
        img_channel = x.shape[-1]

        if data.keypoint_map is not None:
            keypoint_map = data.keypoint_map
            x = np.concatenate([x, keypoint_map], axis=-1)

        x = self.grid_distortion(image=x)["image"]

        data.img = x[:, :, :img_channel]
        if data.keypoint_map is not None:
            data.keypoint_map = x[:, :, img_channel:]

        return data
