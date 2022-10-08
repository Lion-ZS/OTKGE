from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class Fro(Regularizer):
    def __init__(self, weight: float):
        super(Fro, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.norm(f, 2) ** 2
                )
        return norm / factors[0][0].shape[0]


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f) ** 3
                ) / f.shape[0]
        return norm


class L2(Regularizer):
    def __init__(self, weight: float):
        super(L2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f) ** 2
                )
        return norm / factors[0][0].shape[0]
