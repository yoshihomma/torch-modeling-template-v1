import os

import numpy as np
import pydicom
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self, data_dir: str) -> None:
        super().__init__()
        self.data_dir = data_dir
