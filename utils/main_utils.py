"""
Main utils file, all utils functions that are not related to train.
"""

import os
import io

import numpy as np
import torch
import schema
import operator
import functools
import pickle
import cv2
import matplotlib.pyplot as plt
import torchvision

from scipy.interpolate import griddata
from ctypes import *
from PIL import Image
from sys import platform

from torch import nn
from typing import Dict
from utils.types import PathT
from collections import MutableMapping
from utils.config_schema import CFG_SCHEMA
from omegaconf import DictConfig, OmegaConf
from os import path

def reset_data(cfg):

    if cfg['main']['dataset_creation'] == 'pkl':
        if os.path.exists("data/train"):
            dire = "data/train"
            for f in os.listdir(dire):
                os.remove(os.path.join(dire, f))
            dire = "data/validation"
            for f in os.listdir(dire):
                os.remove(os.path.join(dire, f))

    if cfg['main']['dataset_creation'] == 'dump':
        if os.path.exists("data/train.pkl"):
            os.remove("data/train.pkl")
            os.remove("data/validation.pkl")


def get_model_string(model: nn.Module) -> str:
    """
    This function returns a string representing a model (all layers and parameters).
    :param model: instance of a model
    :return: model \n parameters
    """
    model_string: str = str(model)

    n_params = 0
    for w in model.parameters():
        n_params += functools.reduce(operator.mul, w.size(), 1)

    model_string += '\n'
    model_string += f'Params: {n_params}'

    return model_string


def set_seed(seed: int) -> None:
    """
    Sets a seed
    :param seed: seed to set
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dir(path: PathT) -> None:
    """
    Given a path, creating a directory in it
    :param path: string of the path
    """
    if not os.path.exists(path):
        os.mkdir(path)


def warning_print(text: str) -> None:
    """
    This function prints text in yellow to indicate warning
    :param text: text to be printed
    """
    print(f'\033[93m{text}\033[0m')


def validate_input(cfg: DictConfig) -> None:
    """
    Validate the configuration file against schema.
    :param cfg: configuration file to validate
    """
    cfg_types = schema.Schema(CFG_SCHEMA)
    cfg_types.validate(cfg)


def _flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Flatten a dictionary.
    For example:
    {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]} ->
    {'a': 1, 'c_a': 2, 'c_b_x': 5, 'd': [1, 2, 3], 'c_b_y': 10}
    :param d: dictionary to flat
    :param parent_key: key to start from
    :param sep: separator symbol
    :return: flatten dictionary
    """
    items = []

    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def get_flatten_dict(cfg: DictConfig) -> Dict:
    """
    Returns flatten dictionary, given a config dictionary
    :param cfg: config file
    :return: flatten dictionary
    """
    return _flatten_dict(cfg)


def unpack_frame(frame_data, frame_size):
    x, y, z, u, v = [], [], [], [], []
    for i in range(frame_size):
        if i % 5 == 0:
            x.append(frame_data[i])
        if i % 5 == 1:
            y.append(frame_data[i])
        if i % 5 == 2:
            z.append(frame_data[i])
        if i % 5 == 3:
            u.append(frame_data[i])
        if i % 5 == 4:
            v.append(frame_data[i])
    if len(u) < len(x):
        u.append(u[-1])
    if len(v) < len(x):
        v.append(v[-1])
    if len(z) < len(x):
        z.append(z[-1])
    if len(y) < len(x):
        y.append(y[-1])
    return x, y, z, u, v


def load_frame(n_grid, t):
    cwd = os.getcwd()
    grid_path = cwd + f"/preprocessed_data/binary/grid{n_grid}.udp"
    q_path = cwd + f"/preprocessed_data/binary/qsol{t}.udp"
    if platform == "win32":
        clibrary_path = cwd + "/clibrary.dll"
    else:
        clibrary_path = cwd + "/c_code/clibrary.so"
    clibrary = cdll.LoadLibrary(clibrary_path)

    clibrary.process_data.argtypes = [POINTER(c_char), POINTER(c_char), POINTER(POINTER(c_double)), POINTER(c_int),
                                      POINTER(c_int), POINTER(c_int)]
    frame = POINTER(c_double)()
    ni = c_int()
    nj = c_int()
    nk = c_int()

    statues = clibrary.process_data(grid_path.encode("utf-8"), q_path.encode("utf-8"), frame, ni, nj, nk)
    ni = ni.value
    nj = nj.value
    nk = nk.value
    return frame, ni, nj, nk


def find_separation(x, y, u, wall_size):
    separation_upper = 0
    separation_lower = 0
    for i in torch.arange(wall_size, 1.5 * wall_size):
        i = int(i)
        if (u[i] * u[i + 1] < 0) & (u[i] > 0):
            separation_lower = torch.atan(torch.tensor(y[i]) / torch.tensor(x[i]))
            break
    for i in torch.arange(1.5 * wall_size, 2 * wall_size):
        i = int(i)
        if (u[i] * u[i + 1] < 0) & (u[i] > 0):
            separation_upper = torch.atan(torch.tensor(y[i]) / torch.tensor(x[i]))
            break
    return torch.tensor([separation_upper, separation_lower])


def xy2thetar_n(x, y):
    r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
    theta = torch.atan2(y, x)
    # r_n = 1 - torch.exp(-r)
    return r, theta


def imcreate(r, theta, velocity, cfg):
    res = cfg['model']['res']
    image = torch.zeros([res, res])
    eps = 0.0001
    r_max = torch.max(r)
    r_min = torch.min(r)
    theta_max = torch.max(theta)
    theta_min = torch.min(theta)

    points = []
    v_filt = []
    r_filt = []
    for i in torch.arange(len(r)):
        if r[i] > r_max - eps:
            continue
        else:
            points.append((r[i], theta[i]))
            v_filt.append(velocity[i])
            r_filt.append((r[i]))

    r_max = torch.max(torch.tensor(r_filt))
    res_small = res

    kernel_size = cfg['model']['conv_params']['kernel_size']
    padding = cfg['model']['conv_params']['padding']
    stride = cfg['model']['conv_params']['stride']
    for layer_i in np.arange(cfg['model']['num_hid_cnn']):
        res_small = (res_small - kernel_size + 2 * padding) / stride + 1

    grid_r = torch.linspace(r_min, r_max, int(res))
    grid_theta = torch.linspace(theta_min, theta_max, int(res))
    grid_r_small = torch.linspace(r_min, r_max, int(res_small))
    grid_theta_small = torch.linspace(theta_min, theta_max, int(res_small))

    rs, thetas = torch.meshgrid(grid_r, grid_theta)
    rs_small, thetas_small = torch.meshgrid(grid_r_small, grid_theta_small)


    image = griddata(points, v_filt, (rs, thetas), 'linear', fill_value=0)
    image_small = griddata(points, v_filt, (rs_small, thetas_small), 'linear', fill_value=0)


    return torch.tensor(image).type(torch.float), torch.tensor(image_small).type(torch.float)


def load_separations():
    return 0


def load_b():
    return 0


def wall_proc(x, y, u, v, wall_size, train_data, validation_data, counter, sep, future_sep, cfg):
    for i in torch.arange(wall_size, wall_size * 2):
        input_tensor = torch.tensor([x[i - 1], x[i], x[i + 1], y[i - 1], y[i], y[i + 1], u[i - 1], u[i + 1]])
        label_tensor = torch.tensor(u[i])
        if torch.rand(1) < cfg['main']['tt_split']:
            train_data.update({counter[0]: {'label': label_tensor, 'input': input_tensor, 'RANS_label': 0}})
        else:
            validation_data.update({counter[0]: {'label': label_tensor, 'input': input_tensor, 'RANS_label': 0}})
        counter[0] = counter[0] + 1


def image_proc(x, y, u, v, train_data, validation_data, counter, cfg, i, ni, nj, wake):
    velocity = torch.sqrt(torch.tensor([u_i ** 2 + v_i ** 2 for u_i, v_i in zip(u, v)]))
    if wake:
        velocity = velocity < 0.01
    buf = io.BytesIO()
    r, theta = xy2thetar_n(x, y)
    plt.scatter(theta, r, c=velocity)
    plt.savefig(buf, format='png')
    buf.seek(0)
    input_tensor = torch.tensor(plt.imread(buf))
    input_tensor = torch.transpose(input_tensor, 2, 0)
    input_tensor = input_tensor[:3]
    input_tensor = torch.mean(input_tensor, dim=0)
    sep = find_separation(x, y, u, nj)
    if counter[0] < ni:
        future_sep = torch.tensor([0, 0])
    else:
        future_sep = torch.load(f"sep_{i}")
    torch.save(sep, f"temp/sep_{i}")

    label_tensor = future_sep - sep
    if torch.rand(1) < cfg['main']['tt_split']:
        train_data.update({counter[0]: {'label': label_tensor, 'input': input_tensor}})
    else:
        validation_data.update({counter[0]: {'label': label_tensor, 'input': input_tensor}})


def Rey_proc(x, y, u, v, train_data, validation_data, counter, cfg):

    lam = []
    input_tensor = []
    R = torch.tensor([u.size, 3, 3])
    S = torch.tensor([u.size, 3, 3])
    for i, j in zip(R[:][:][0][0]):
        lam[0][i][j] = torch.trace(S @ S)
        lam[1][i][j] = torch.trace(R @ R)
        lam[2][i][j] = torch.trace(S @ S @ S)
        lam[3][i][j] = torch.trace(R @ R @ S)
        lam[4][i][j] = torch.trace(R @ R @ S @ S)
    r, theta = xy2thetar_n(x, y)
    buf = io.BytesIO()
    input_tensor[0], mm = imcreate(r, theta, lam[0], cfg['main']['r_res'], cfg['main']['theta_res'])
    input_tensor[1], mm = imcreate(r, theta, lam[1], cfg['main']['r_res'], cfg['main']['theta_res'])
    input_tensor[2], mm = imcreate(r, theta, lam[2], cfg['main']['r_res'], cfg['main']['theta_res'])
    input_tensor[3], mm = imcreate(r, theta, lam[3], cfg['main']['r_res'], cfg['main']['theta_res'])
    input_tensor[4], mm = imcreate(r, theta, lam[4], cfg['main']['r_res'], cfg['main']['theta_res'])


    label_tensor = load_b()

    if torch.rand(1) < cfg['main']['tt_split']:
        train_data.update({counter[0]: {'label': label_tensor, 'input': input_tensor}})
    else:
        validation_data.update({counter[0]: {'label': label_tensor, 'input': input_tensor}})


def vel_proc(x, y, u, v, train_data, validation_data, counter, cfg, i, ni):
    velocity = torch.sqrt(torch.tensor([u_i ** 2 + v_i ** 2 for u_i, v_i in zip(u, v)]))
    r, theta = xy2thetar_n(x, y)
    input_tensor, input_tensor_small = imcreate(r, theta, velocity, cfg)
    input_tensor = torch.unsqueeze(input_tensor, dim=0)
    input_tensor_small = torch.unsqueeze(input_tensor_small, dim=0)

    if counter[0] < 1:
        label_tensor = torch.zeros_like(input_tensor_small)

    else:
        label_tensor = torch.load(f"temp/img_{i}")
    torch.save(input_tensor_small, f"temp/img_{i}")

    if torch.rand(1) < cfg['main']['tt_split']:
        train_data.update({counter[0]: {'label': label_tensor, 'input': input_tensor}})
    else:
        validation_data.update({counter[0]: {'label': label_tensor, 'input': input_tensor}})


def create_dataset(x, y, u, v, cfg, counter, train_handle, validation_handle, i, ni, nj, train_data, validation_data):

    x = torch.tensor(x)
    y = torch.tensor(y)
    u = torch.tensor(u)
    v = torch.tensor(v)

    if cfg['main']['exp_type'] == 'wall':
        wall_proc(x, y, u, v, nj, train_data, validation_data, counter, cfg)
    if cfg['main']['exp_type'] == 'image':
        image_proc(x, y, u, v, train_data, validation_data, counter, cfg, i, ni, nj, False)
    if cfg['main']['exp_type'] == 'wake':
        image_proc(x, y, u, v, train_data, validation_data, counter, cfg, i, ni, nj, True)
    if cfg['main']['exp_type'] == 'Rey':
        Rey_proc(x, y, u, v, train_data, validation_data, counter, cfg)
    if cfg['main']['exp_type'] == 'vel':
        vel_proc(x, y, u, v, train_data, validation_data, counter, cfg, i, ni)

    counter[0] = counter[0] + 1

    if cfg['main']['dataset_creation'] == 'dump':
        dump_every = 30
        if counter[0] % dump_every == 0:
            if len(train_data) != 0:
                pickle.dump(train_data, train_handle)
                # torch.save(train_data, train_handle)
            if len(validation_data) != 0:
                pickle.dump(validation_data, validation_handle)
                # torch.save(validation_data, validation_handle)

            train_data = {}
            validation_data = {}

    if cfg['main']['dataset_creation'] == 'pkl':
        if len(train_data) != 0:
            torch.save(train_data, f"data/train/train_{counter[0]}.pkl")
        if len(validation_data) != 0:
            torch.save(validation_data, f"data/validation/validation_{counter[0]}.pkl")


    if cfg['main']['dataset_creation'] == 'image':
        if len(train_data) != 0:
            Image.save(Image.open(train_data[counter[0]]), f"temp/train_{counter[0]}")
        if len(validation_data) != 0:
            Image.save(Image.open(validation_data[counter[0]]), f"temp/validation_{counter[0]}")
        train_data = {}
        validation_data = {}


def init(cfg: DictConfig) -> None:
    """
    :cfg: hydra configuration file
    """
    # TODO: Trains
    validate_input(cfg)
