from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class DataContainer:
    img: np.ndarray
    label: Optional[np.ndarray] = None


@dataclass
class BatchDataContainer:
    img: torch.Tensor
    label: Optional[torch.Tensor] = None
