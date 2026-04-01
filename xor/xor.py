import torch

import torch.nn as nn
import torch.optim as optim
from torch import Tensor


class XOR(nn.Models):
    def __init__(self):
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = nn.ReLU()
        x = self.linear2(x)
        x = nn.Sigmoid()
        
        return x