from typing import Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from ...data.collate_batch import collate_batch
from ...data.datasets import PretrainKeypointDataset
from ...data.transforms import Compose
from ...utils.yacs import CfgNode
from .common import split_df_train_and_valid


def setup_dataset(
    cfg: CfgNode,
    data_dir: str,
    train_transforms: Compose,
    valid_transforms: Optional[Compose] = None,
) -> Tuple[PretrainKeypointDataset, Optional[PretrainKeypointDataset]]:
    assert cfg.dataset.type == "PretrainKeypointDataset"

    df = pd.read_csv(cfg.dataset.keypoint_csv)

    train_df, valid_df = split_df_train_and_valid(cfg, df, "filename")

    def _parse_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["source", "filename", "level"]).reset_index(drop=True)
        df["filename"] = df["filename"].str.replace(".jpg", ".npy")
        df["series_id"] = df["source"] + "_" + df["filename"].str.split(".").str[0]
        return df

    train_df = _parse_df(train_df)
    valid_df = _parse_df(valid_df) if valid_df is not None else None

    train_datset = PretrainKeypointDataset(
        data_dir=data_dir,
        df=train_df,
        num_instances=cfg.dataset.num_instances,
        transforms=train_transforms,
    )
    if valid_df is not None:
        valid_dataset = PretrainKeypointDataset(
            data_dir=data_dir,
            df=valid_df,
            num_instances=cfg.dataset.num_instances,
            transforms=valid_transforms,
        )
    else:
        valid_dataset = None

    return train_datset, valid_dataset


class CustomDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_batch,
        )

        self._dataset = dataset


def setup_data_loader(
    cfg: CfgNode,
    data_dir: str,
    train_transforms: Compose,
    valid_transforms: Optional[Compose] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_dataset, valid_dataset = setup_dataset(
        cfg, data_dir, train_transforms, valid_transforms
    )

    train_data_loader = DataLoader(
        train_dataset,
        cfg.solver.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    if valid_dataset is not None:
        assert valid_transforms is not None
        valid_data_loader = DataLoader(
            valid_dataset,
            cfg.solver.valid_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    else:
        valid_data_loader = None
    return train_data_loader, valid_data_loader
