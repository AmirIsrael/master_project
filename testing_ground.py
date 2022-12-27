import random

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from models.MY_MODEL import MyModel
from scipy.interpolate import griddata
from os import path
from utils import main_utils, train_utils, model_utils


def xy2thetar_n(x, y):
    r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
    theta = torch.atan2(y, x)
    # r_n = 1 - torch.exp(-r)
    return r, theta


def imcreate(r, theta, velocity, r_res, theta_res, cfg):
    eps = 0.0001
    r_max = torch.max(r)
    r_min = torch.min(r)
    theta_max = torch.max(theta)
    theta_min = torch.min(theta)

    points = []
    v_filt = []
    r_filt = []
    for i in torch.arange(r):
        if r[i] > r_max - eps:
            continue
        else:
            points.append((r[i], theta[i]))
            v_filt.append(velocity[i])
            r_filt.append((r[i]))

    r_max = torch.max(torch.tensor(r_filt))
    r_res_small = r_res
    theta_res_small = theta_res

    kernel_size = cfg['model']['conv_params']['kernel_size']
    padding = cfg['model']['conv_params']['padding']
    stride = cfg['model']['conv_params']['stride']
    for layer_i in torch.arange(cfg['model']['num_hidden_cnn']):
        r_res_small = (r_res_small - kernel_size + 2 * padding) / stride + 1
        theta_res_small = (theta_res_small - kernel_size + 2 * padding) / stride + 1

    grid_r = torch.linspace(r_min, r_max, r_res)
    grid_theta = torch.linspace(theta_min, theta_max, theta_res)
    grid_r_small = torch.linspace(r_min, r_max, r_res_small)
    grid_theta_small = torch.linspace(theta_min, theta_max, theta_res_small)

    rs, thetas = torch.meshgrid(grid_r, grid_theta)
    rs_small, thetas_small = torch.meshgrid(grid_r_small, grid_theta_small)

    image = griddata(points, v_filt, (rs, thetas), 'linear', fill_value=0)
    image_small = griddata(points, v_filt, (rs_small, thetas_small), 'linear', fill_value=0)

    return torch.tensor(image).type(torch.float), torch.tensor(image_small).type(torch.float)


# f = open("data/train/train_2.pkl", "rb")
# train = torch.load(f)
# img = train[1]['label']
# img = torch.squeeze(img)
# plt.imshow(img, cmap='gray')
# plt.show()


log_file_path = path.join(path.dirname(path.abspath(__file__)), 'config', 'config.yaml')
f = open(log_file_path)
cfg = yaml.full_load(f)
model_params = model_utils.get_model_params(cfg)

model = MyModel(model_params, in_dim=1, out_dim=1, in_classes=0, out_classes=0)
# model.load_state_dict(torch.load("saved_models/model.pt", map_location='cpu'))
# img = torch.rand([1,500,500])
# img = torch.squeeze(img)
#
# plt.imshow(img, cmap='gray')
# plt.show()
#
# img = torch.unsqueeze(img, dim=0)
# out = model(img)
# out = torch.squeeze(out).detach().numpy()
# plt.imshow(out, cmap='gray')
# plt.show()

print(model)

with torch.no_grad():
    img = torch.randn(1, 500, 500)
    out = model(img)
    print(out.mean())
    print(out.std())
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.show()

    model.load_state_dict(torch.load("saved_models/model.pt", map_location='cpu'))
    out = model(img)
    print(out.mean())
    print(out.std())
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.show()
