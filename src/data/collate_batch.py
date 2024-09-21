from typing import List

import torch

from .data_container import AxisBatchDataContainer, BatchDataContainer


def collate_batch(batch: List[BatchDataContainer]) -> BatchDataContainer:
    """
    Collate function for the batch data container
    """

    img = torch.stack([b.img for b in batch])
    label = (
        torch.stack([b.label for b in batch]) if batch[0].label is not None else None
    )
    return BatchDataContainer(
        img=img,
        label=label,
    )
