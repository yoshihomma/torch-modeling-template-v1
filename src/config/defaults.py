from ..utils.yacs import CfgNode as CN


# TODO: Implement get_default_confg function
# https://www.kaggle.com/code/awsaf49/birdclef23-pretraining-is-all-you-need-train/notebook


def get_default_confg() -> CN:
    _C = CN()
    _C.name = ""  # this variable is automatically stored
    _C.cfg_path = ""  # this variable is automatically stored
    _C.git_hash = ""  # this variable is automatically stored
    _C.seed = 2024
    _C.cudnn_benchmark = True
    _C.num_workers = 2
    _C.pin_memory = True
    _C.precision = 32
    _C.save_model_policy = "best_valid"  # "best_valid" or "last_epoch"

    # --------------------
    # Dataset
    # --------------------
    _C.dataset = CN()
    _C.dataset.type = ""
    _C.dataset.cv_policy = "filter"
    _C.dataset.cv_num_folds = 5
    _C.dataset.cv_valid_fold_index = 0
    _C.dataset.use_fix_fold = False

    # --------------------
    # Transform
    # --------------------
    _C.transforms = CN()
    _C.transforms.train = []  # e.g. ["Resize", ...]
    _C.transforms.valid = []

    # --------------------
    # Model
    # --------------------
    _C.model = CN()
    _C.model.type = ""
    _C.model.backbone = ""
    _C.model.pretrained_model = "true"

    # --------------------
    # LightningModel
    # --------------------
    _C.lightning_model = CN()
    _C.lightning_model.type = "PretrainKeypointLightningModel"
    # --------------------
    # Loss
    # --------------------
    _C.loss = CN()
    _C.loss.type = ""
    _C.loss.reduction = "mean"
    # --------------------
    # Solver
    # --------------------
    _C.solver = CN()
    _C.solver.train_batch_size = 32
    _C.solver.valid_batch_size = 32
    # value <= 0 means not using early stopping
    _C.solver.early_stopping_patience = 10
    _C.solver.validation_monitor = "valid_loss"
    _C.solver.optimizer = "Adam"
    _C.solver.lr_scheduler = "CosineAnnealingLR"
    _C.solver.max_epochs = 50
    _C.solver.base_lr = 0.001
    _C.solver.weight_decay = 1e-6
    _C.solver.warmup_epochs = 1
    _C.solver.warmup_start_lr = 1e-7
    # not using parameters
    _C.solver.momentum = 0.9

    return _C
