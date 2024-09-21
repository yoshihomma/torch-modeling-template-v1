from albumentations import Normalize as AlbumenationsNormalize

from ..data_container import DataContainer
from .transform_base import TransformBase


class AlbuNormalize(TransformBase):
    def __init__(self, mean: float, std: float):
        super().__init__(True, 1.0)
        self.norm = AlbumenationsNormalize(mean=mean, std=std, p=1.0, always_apply=True)

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.img is not None
        x = data.img
        x = self.norm(image=x)["image"]
        data.img = x
        return data
