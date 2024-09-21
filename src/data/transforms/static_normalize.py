from ..data_container import DataContainer
from .transform_base import TransformBase


class StaticNormalize(TransformBase):
    def __init__(self, div_value: float = 255.0):
        super().__init__(True, 1.0)
        self.div_value = div_value

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.img is not None
        x = data.img
        data.img = x / self.div_value
        return data
