from albumentations.augmentations.dropout import \
    CoarseDropout as AlbumentationsCoarseDropout

from ..data_container import DataContainer
from .transform_base import TransformBase


class CoarseDropout(TransformBase):
    def __init__(
        self,
        max_holes: int,
        min_holes: int,
        max_width: int,
        min_width: int,
        max_height: int,
        min_height: int,
        always_apply: bool,
        p: float,
    ) -> None:
        super().__init__(always_apply=True, p=1.0)

        self.coarse_dropout = AlbumentationsCoarseDropout(
            max_holes=max_holes,
            min_holes=min_holes,
            max_width=max_width,
            min_width=min_width,
            max_height=max_height,
            min_height=min_height,
            always_apply=always_apply,
            p=p,
        )

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.img is not None

        data.img = self.coarse_dropout(image=data.img)["image"]
        return data
