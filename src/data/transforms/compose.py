from typing import List

from ..data_container import DataContainer
from .transform_base import TransformBase


class Compose:
    def __init__(self, transforms: List[TransformBase]) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, data: DataContainer) -> DataContainer:
        return self.apply(data)

    def apply(self, data: DataContainer) -> DataContainer:
        x = data
        for t in self.transforms:
            if t.is_apply():
                x = t.apply(data)
        return x
