from ..utils.yacs import CfgNode as CN

_C = CN()
_C.transforms = CN()
# Resize
_C.transforms.resize = CN()
_C.transforms.resize.width = 256
_C.transforms.resize.height = 256
# Normalize
_C.transforms.normalize = CN()
_C.transforms.normalize.eps = 1e-6
# ShiftScaleRotate
_C.transforms.shift_scale_rotate = CN()
_C.transforms.shift_scale_rotate.shift_limit = 0.0625
_C.transforms.shift_scale_rotate.scale_limit = 0.05
_C.transforms.shift_scale_rotate.rotate_limit = 10
_C.transforms.shift_scale_rotate.always_apply = False
_C.transforms.shift_scale_rotate.p = 0.5
# CoarseDropout
_C.transforms.coarse_dropout = CN()
_C.transforms.coarse_dropout.max_holes = 16
_C.transforms.coarse_dropout.min_holes = 1
_C.transforms.coarse_dropout.max_width = 32
_C.transforms.coarse_dropout.min_width = 1
_C.transforms.coarse_dropout.max_height = 32
_C.transforms.coarse_dropout.min_height = 1
_C.transforms.coarse_dropout.always_apply = False
_C.transforms.coarse_dropout.p = 0.5
# ElasticTransform
_C.transforms.elastic_transform = CN()
_C.transforms.elastic_transform.alpha = 1
_C.transforms.elastic_transform.sigma = 50
_C.transforms.elastic_transform.alpha_affine = 50
# GridDistortion
_C.transforms.grid_distortion = CN()
_C.transforms.grid_distortion.num_steps = 5
_C.transforms.grid_distortion.distort_limit = 0.05
_C.transforms.grid_distortion.always_apply = False
_C.transforms.grid_distortion.p = 0.5
# HorizontalFlip
_C.transforms.horizontal_flip = CN()
_C.transforms.horizontal_flip.always_apply = False
_C.transforms.horizontal_flip.p = 0.5
# VerticalFlip
_C.transforms.vertical_flip = CN()
_C.transforms.vertical_flip.always_apply = False
_C.transforms.vertical_flip.p = 0.5
# RandomBrightnessContrast
_C.transforms.random_brightness_contrast = CN()
_C.transforms.random_brightness_contrast.brightness_limit = 0.2
_C.transforms.random_brightness_contrast.contrast_limit = 0.2
_C.transforms.random_brightness_contrast.always_apply = False
_C.transforms.random_brightness_contrast.p = 0.5
# MotionBlur
_C.transforms.motion_blur = CN()
_C.transforms.motion_blur.blur_limit = 3
_C.transforms.motion_blur.always_apply = False
_C.transforms.motion_blur.p = 0.5
# MotionBlur
_C.transforms.median_blur = CN()
_C.transforms.median_blur.blur_limit = 3
_C.transforms.median_blur.always_apply = False
_C.transforms.median_blur.p = 0.5
# GaussianBlur
_C.transforms.gaussian_blur = CN()
_C.transforms.gaussian_blur.blur_limit = 3
_C.transforms.gaussian_blur.always_apply = False
_C.transforms.gaussian_blur.p = 0.5
# GaussianNoise
_C.transforms.gaussian_noise = CN()
_C.transforms.gaussian_noise.var_limit = (3.0, 9.0)
_C.transforms.gaussian_noise.always_apply = False
_C.transforms.gaussian_noise.p = 0.5
# OpticalDistortion
_C.transforms.optial_distortion = CN()
_C.transforms.optial_distortion.distort_limit = 1.0
_C.transforms.optial_distortion.always_apply = False
_C.transforms.optial_distortion.p = 0.5
# Cut
_C.transforms.cut = CN()
_C.transforms.cut.width_region = (0.25, 0.75)
_C.transforms.cut.height_region = (0.25, 0.75)
# AlbuNormalize
_C.transforms.albu_normalize = CN()
_C.transforms.albu_normalize.mean = 0.5
_C.transforms.albu_normalize.std = 0.5
