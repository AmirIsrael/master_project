import torch.nn as nn
from torch import Tensor
from typing import Sequence
from typing import Union


class MLP(nn.Module):

    def __init__(
            self, dims: Sequence[int]
    ):
        super().__init__()
        last_dim = dims[0]
        layers = []
        nonlin = nn.LeakyReLU()
        for i, dim in enumerate(dims):
            if i == len(dims) - 1:
                layers.extend([nn.Linear(last_dim, dim)])
                break
            layers.extend([nn.Linear(last_dim, dim), nonlin])
            last_dim = dim
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:

        return self.model(x)
