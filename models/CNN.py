import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Sequence
from models.MLP import MLP


class CNN(nn.Module):

    def __init__(
            self,
            out_classes: int,
            channels: Sequence[int],
            pool_every: int,
            exp_type: str,
            conv_params: dict,
            image_size: Sequence[int],
                            ):

        super().__init__()
        assert channels

        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.conv_params = conv_params
        self.exp_type = exp_type
        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()
        self.image_size = image_size

    def _make_feature_extractor(self):
        layers = []
        P = self.pool_every
        N = len(self.channels) - 1
        t_channels = self.channels
        for i in range(N):
            layers.extend([nn.Conv2d(t_channels[i], t_channels[i + 1], *self.conv_params.values()), nn.LeakyReLU()])
            # if (i + 1) % P == 0:
            #     layers.extend(nn.MaxPool2d())
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the classifier part.
        :return: Number of features.
        """
        rng_state = torch.get_rng_state()
        try:
            in_size = self.image_size
            x = torch.rand(size=in_size)
            x = x.unsqueeze(0)
            return np.prod(tuple(self.feature_extractor(x).size()))
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        mlp = None
        if self.out_classes != 0:
            mlp = MLP([self._n_features(), self._n_features(), self.out_classes])
        return mlp

    def forward(self, x: Tensor):

        out: Tensor = None

        features = self.feature_extractor(x)
        if self.out_classes == 0:
            return features
        else:
            features = features.view(features.size(0), -1)
            out = self.mlp(features)
            return out


        #
        # features = self.feature_extractor(x)
        # mlp = MLP([])

