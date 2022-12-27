"""
Main file
We will run the whole program from here
"""
import os
from os import path

import torch
import yaml
import pickle
import wandb

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset import MyDataset
from models.MY_MODEL import MyModel
from train import train
from utils import main_utils, train_utils, model_utils
from utils.train_logger import TrainLogger

torch.backends.cudnn.benchmark = True


def main() -> None:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
    log_file_path = path.join(path.dirname(path.abspath(__file__)), 'config', 'config.yaml')
    f = open(log_file_path)
    cfg = yaml.full_load(f)
    main_utils.init(cfg)
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])
    # create datasets

    # init wandb
    wandb.init(project=cfg['main']['exp_type'], config=cfg)

    if cfg['main']['load_dataset'] is True:
        delta_t = 1
        time_steps = cfg['main']['time_steps']
        time_steps = int(time_steps / 10)

        main_utils.reset_data(cfg)

        counter = [0]
        n_grid = 10
        train_data = {}
        validation_data = {}
        with open('data/train.pkl', 'wb') as train_handle:
            with open('data/validation.pkl', 'wb') as validation_handle:
                for t in torch.arange(time_steps, 1, -delta_t):
                    frame_data, ni, nj, nk = main_utils.load_frame(n_grid, t * 10)
                    x, z, y, u, v = main_utils.unpack_frame(frame_data, ni * nj * nk)
                    for i in torch.arange(ni):
                        if i == round(ni/2):
                            if cfg['main']['dataset_creation'] != 'dump':
                                train_data = {}
                                validation_data = {}
                            x_z = x[i::ni]
                            y_z = y[i::ni]
                            u_z = u[i::ni]
                            v_z = v[i::ni]
                            main_utils.create_dataset(x_z, y_z, u_z, v_z, cfg, counter, train_handle,
                                                  validation_handle, i, ni, nj, train_data, validation_data)

                if cfg['main']['dataset_creation'] == 'dump':
                    if len(train_data) != 0:
                        pickle.dump(train_data, train_handle)
                        # torch.save(train_data, train_handle)
                    if len(validation_data) != 0:
                        pickle.dump(validation_data, validation_handle)
                        # torch.save(validation_data, validation_handle)

    train_dataset = MyDataset(path=cfg['main']['paths']['train'])
    val_dataset = MyDataset(path=cfg['main']['paths']['validation'])

    train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['main']['num_workers'])
    eval_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=True,
                             num_workers=cfg['main']['num_workers'])

    # Init model
    model_params = model_utils.get_model_params(cfg)
    if cfg['main']['exp_type'] == 'wall':
        model = MyModel(model_params, in_dim=0, out_dim=0, in_classes=8, out_classes=2)
    if cfg['main']['exp_type'] == 'image':
        model = MyModel(model_params, in_dim=1, out_dim=5, in_classes=0, out_classes=2)
    if cfg['main']['exp_type'] == 'wake':
        model = MyModel(model_params, in_dim=1, out_dim=5, in_classes=0, out_classes=2)
    if cfg['main']['exp_type'] == 'Rey':
        model = MyModel(model_params, in_dim=5, out_dim=5, in_classes=0, out_classes=0)
    if cfg['main']['exp_type'] == 'vel':
        model = MyModel(model_params, in_dim=1, out_dim=1, in_classes=0, out_classes=0)

    # TODO: Add gpus_to_use
    if cfg['main']['parallel']:
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    logger.write(main_utils.get_model_string(model))

    # Run model
    train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    metrics = train(model, train_loader, eval_loader, train_params, logger)
    hyper_parameters = main_utils.get_flatten_dict(cfg['train'])

    logger.report_metrics_hyper_params(hyper_parameters, metrics)

    torch.save(model.state_dict(), "saved_models/model.pt")

    art = wandb.Artifact("my-object-detector", type="model")
    art.add_file("saved_models/model.pt")
    wandb.log_artifact(art)
    wandb.finish()

if __name__ == '__main__':
    main()
