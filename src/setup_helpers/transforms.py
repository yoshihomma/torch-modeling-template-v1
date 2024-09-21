from typing import Any, Dict, List, Optional

from ..data import transforms as T
from ..data.transforms.transform_base import TransformBase
from ..utils.yacs import CfgNode


# TODO: Implement get_transform function
def _get_transform(
    cfg: CfgNode, t: str, overwrite_param: Optional[Dict[str, Any]] = None
) -> TransformBase:
    if t == "Resize":
        param = cfg.transforms.resize
        transform_class = T.Resize
    elif t == "Normalize":
        param = cfg.transforms.normalize
        transform_class = T.Normalize
    elif t == "ShiftScaleRotate":
        param = cfg.transforms.shift_scale_rotate
        transform_class = T.ShiftScaleRotate
    elif t == "CoarseDropout":
        param = cfg.transforms.coarse_dropout
        transform_class = T.CoarseDropout
    elif t == "ElasticTransform":
        param = cfg.transforms.elastic_transform
        transform_class = T.ElasticTransform
    elif t == "GridDistortion":
        param = cfg.transforms.grid_distortion
        transform_class = T.GridDistortion
    elif t == "HorizontalFlip":
        param = cfg.transforms.horizontal_flip
        transform_class = T.HorizontalFlip
    elif t == "VerticalFlip":
        param = cfg.transforms.vertical_flip
        transform_class = T.VerticalFlip
    elif t == "RandomBrightnessContrast":
        param = cfg.transforms.random_brightness_contrast
        transform_class = T.RandomBrightnessContrast
    elif t == "MotionBlur":
        param = cfg.transforms.motion_blur
        transform_class = T.MotionBlur
    elif t == "MedianBlur":
        param = cfg.transforms.median_blur
        transform_class = T.MedianBlur
    elif t == "GaussianBlur":
        param = cfg.transforms.gaussian_blur
        transform_class = T.GaussianBlur
    elif t == "GaussianNoise":
        param = cfg.transforms.gaussian_noise
        transform_class = T.GaussianNoise
    elif t == "OpticalDistortion":
        param = cfg.transforms.optial_distortion
        transform_class = T.OpticalDistortion
    elif t == "Cut":
        param = cfg.transforms.cut
        transform_class = T.Cut
    elif t == "AlbuNormalize":
        param = cfg.transforms.albu_normalize
        transform_class = T.AlbuNormalize
    elif t == "StaticNormalize":
        param = cfg.transforms.static_normalize
        transform_class = T.StaticNormalize
    else:
        raise ValueError(f"Unexpected transform: {t}")

    if overwrite_param:
        param.update(overwrite_param)
    transform = transform_class(**param)
    return transform


def setup_transforms(cfg: CfgNode, mode: str) -> T.Compose:
    cfg_transforms: List[str] = []
    cfg_transforms = cfg.transforms[mode]

    transforms = []

    for t in cfg_transforms:
        if type(t) is str:
            transform = _get_transform(cfg, t)
        elif type(t) is dict:
            assert len(t) == 1
            # when OneOf
            if (transform_type := list(t.keys())[0]) == "OneOf":
                assert "transforms" in t["OneOf"].keys()
                _ts = []
                for _t in t["OneOf"]["transforms"]:
                    if type(_t) is str:
                        _t = _get_transform(cfg, _t)
                    # when OneOf with overwrite parameter
                    elif type(_t) is dict:
                        assert len(_t) == 1
                        _transform_type = list(_t.keys())[0]
                        # not support nested OneOf
                        assert _transform_type != "OneOf"
                        _overwrite_param = _t[_transform_type]
                        _t = _get_transform(cfg, _transform_type, _overwrite_param)
                    else:
                        raise ValueError()
                    _ts.append(_t)
                # _ts = [get_transform(cfg, _t, mode) for _t in t["OneOf"]["transforms"]]
                transform = T.OneOf(t["OneOf"]["always_apply"], t["OneOf"]["p"], _ts)
            # when overwrite parameter (not OneOf)
            else:
                overwrite_param = t[transform_type]
                transform = _get_transform(cfg, transform_type, overwrite_param)
        else:
            raise ValueError()
        transforms.append(transform)

    transforms = T.Compose(transforms)
    return transforms
