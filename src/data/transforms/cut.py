from typing import Tuple

from ..data_container import DataContainer
from .transform_base import TransformBase


class Cut(TransformBase):
    def __init__(
        self, width_region: Tuple[float, float], height_region: Tuple[float, float]
    ) -> None:
        super().__init__(always_apply=True, p=1.0)

        self.width_region = width_region
        self.height_region = height_region

    def apply(self, data: DataContainer) -> DataContainer:
        org_height, org_width = data.img.shape[:2]
        min_height = int(org_height * self.height_region[0])
        max_height = int(org_height * self.height_region[1])
        min_width = int(org_width * self.width_region[0])
        max_width = int(org_width * self.width_region[1])

        data.img = data.img[min_height:max_height, min_width:max_width]

        if data.keypoint is not None:
            keypoint = data.keypoint.copy()
            keypoint[:, 0] -= min_width
            keypoint[:, 1] -= min_height
            data.keypoint = keypoint

        if data.keypoint_map is not None:
            data.keypoint_map = data.keypoint_map[
                min_height:max_height, min_width:max_width
            ]

        return data
