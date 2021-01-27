import math
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, Optional


class LSN(nn.Module):
    
    """ Custom Linear layer that modifies standard ReLU layer"""
    
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, scale:int = 20000, inplace:bool = False):
        super(LSN, self).__init__()
        self.inplace = inplace
        self.scale = scale

    def forward(self, input: Tensor) -> Tensor:
        y_relu = F.relu(input, inplace=self.inplace)
        num = y_relu * self.scale;
        denom = torch.sum(y_relu);
        return num/(denom + 1e-8)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
