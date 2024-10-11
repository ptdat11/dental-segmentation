import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from abc import ABC


class BaseModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pass
