"""
    Example for a simple model
"""
import torch
from torch import nn, Tensor
from utils.model_utils import ModelParams
from models.MLP import MLP
from models.CNN import CNN


def find_dims(model_params, in_dim, out_dim):
    dims = []
    s_len = 3
    for i, dim_i in enumerate(torch.arange(model_params.num_hid_mlp)):
        if i == model_params.num_hid_mlp - 1:
            dims.append(out_dim)
            break
        if i >= model_params.num_hid_mlp - s_len:
            dims.append(dims[i - 1] - in_dim)
            continue
        if i < s_len:
            dims.append(in_dim * (i + 1))
            continue

        dims.append(dims[i - 1])
    return dims


def find_channels(model_params, in_channel, last_dim):
    dims = []
    s_len = 5
    for i, dim_i in enumerate(torch.arange(model_params.num_hid_cnn)):
        if i == model_params.num_hid_cnn - 1:
            dims.append(last_dim)
            break
        if i >= model_params.num_hid_cnn - s_len:
            dims.append(dims[i - 1] - in_channel)
            continue
        if i < s_len:
            dims.append(in_channel * (i + 1))
            continue

        dims.append(dims[i - 1])
    return dims


class MyModel(nn.Module):
    """
    Example for a simple model
    """

    def __init__(self, model_params: ModelParams, in_dim: int, out_dim: int, in_classes: int, out_classes: int):
        super(MyModel, self).__init__()
        if model_params.exp_type == 'wall':
            dims = find_dims(model_params, in_classes, out_classes)
            self.classifier = MLP(dims=dims)
        if model_params.exp_type == 'image':
            channels = find_channels(model_params, in_dim, out_dim)
            self.classifier = CNN(out_classes=out_classes, channels=channels, pool_every=model_params.pooling,
                                  conv_params=model_params.conv_params, exp_type=model_params.exp_type)
        if model_params.exp_type == 'wake':
            channels = find_channels(model_params, in_dim, out_dim)
            self.classifier = CNN(out_classes=out_classes, channels=channels, pool_every=model_params.pooling,
                                  conv_params=model_params.conv_params, exp_type=model_params.exp_type)
        if model_params.exp_type == 'Rey':
            channels = find_channels(model_params, in_dim, out_dim)
            self.classifier = CNN(out_classes=out_classes, channels=channels, pool_every=model_params.pooling,
                                  conv_params=model_params.conv_params, exp_type=model_params.exp_type)
        if model_params.exp_type == 'vel':
            channels = find_channels(model_params, in_dim, out_dim)
            self.classifier = CNN(out_classes=out_classes, channels=channels, pool_every=model_params.pooling,
                                  conv_params=model_params.conv_params, exp_type=model_params.exp_type, image_size=[model_params.res, model_params.res])


    def forward(self, x: Tensor) -> Tensor:
        """
        Forward x through MyModel
        :param x:
        :return:
        """
        return self.classifier(x)
