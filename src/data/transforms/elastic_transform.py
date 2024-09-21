import numpy as np
from albumentations.augmentations.geometric import \
    ElasticTransform as AlbumentationsElasticTransform

from ..data_container import DataContainer
from .transform_base import TransformBase


class ElasticTransform(TransformBase):
    def __init__(
        self,
        alpha: int,
        sigma: int,
        alpha_affine: int,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)

        self.elastic_transform = AlbumentationsElasticTransform(
            alpha=alpha,
            sigma=sigma,
            alpha_affine=alpha_affine,
            always_apply=True,
            p=1.0,
        )

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.img is not None

        x = data.img
        img_channel = x.shape[-1]

        if data.keypoint_map is not None:
            keypoint_map = data.keypoint_map
            x = np.concatenate([x, keypoint_map], axis=-1)

        x = self.elastic_transform(image=x)["image"]

        data.img = x[:, :, :img_channel]
        if data.keypoint_map is not None:
            data.keypoint_map = x[:, :, img_channel:]

        return data
