import numpy as np

from ..data_container import DataContainer
from .transform_base import TransformBase


class Normalize(TransformBase):
    def __init__(self, eps=1e-6):
        super().__init__(True, 1.0)
        self.eps = eps

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.img is not None
        x = data.img
        h, w, c = x.shape
        mean = np.mean(x.reshape(h * w, c), axis=0)
        std = np.std(x.reshape(h * w, c), axis=-0)

        x = (x - mean) / (std + self.eps)
        data.img = x
        return data
