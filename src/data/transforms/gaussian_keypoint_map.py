import numpy as np

from ..data_container import DataContainer
from .transform_base import TransformBase


class GaussanKeypointMap(TransformBase):
    def __init__(self, sigma: float) -> None:
        super().__init__(always_apply=True, p=1.0)

        self.sigma = sigma

    def apply(self, data: DataContainer) -> DataContainer:
        assert data.keypoint is not None

        height, width = data.img.shape[:2]

        cord_map = np.array(
            [(y, x) for y in range(height) for x in range(width)]
        ).reshape((height, width, 2))

        if len(data.keypoint.shape) == 2:
            assert data.level_id is not None
            keypoint_map = np.zeros((height, width, 5), dtype=np.float32)
            for level_id, keypoint in zip(data.level_id, data.keypoint):
                x, y = keypoint[:2]
                distance_map = cord_map - np.array((y, x))
                map = self._calc_gaussian(np.sum(np.square(distance_map), axis=-1))
                keypoint_map[:, :, level_id] = map
        elif len(data.keypoint.shape) == 1:
            keypoint_map = np.zeros((height, width, 1), dtype=np.float32)
            x, y = data.keypoint
            distance_map = cord_map - np.array((y, x))
            map = self._calc_gaussian(np.sum(np.square(distance_map), axis=-1))
            keypoint_map[:, :, 0] = map
        else:
            raise ValueError(f"Invalid keypoint shape: {data.keypoint.shape}")

        data.keypoint_map = keypoint_map
        return data

    def _calc_gaussian(self, x: float) -> float:
        return np.exp(-1 * np.square(x) / np.square(self.sigma))
