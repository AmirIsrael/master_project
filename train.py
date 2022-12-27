"""
Here, we will run everything that is related to the training procedure.
"""

import time

import numpy as np
import torch
import wandb
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import train_utils
from torch.utils.data import DataLoader
from utils.types import Scores, Metrics
from utils.train_utils import TrainParams
from utils.train_logger import TrainLogger


def get_metrics(best_eval_score: float, eval_score: float, train_loss: float) -> Metrics:
    """
    Example of metrics dictionary to be reported to tensorboard. Change it to your metrics
    :param best_eval_score:
    :param eval_score:
    :param train_loss:
    :return:
    """
    return {'Metrics/BestAccuracy': best_eval_score,
            'Metrics/LastAccuracy': eval_score,
            'Metrics/LastLoss': train_loss}


def train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, train_params: TrainParams,
          logger: TrainLogger) -> Metrics:
    """
    Training procedure. Change each part if needed (optimizer, loss, etc.)
    :param model:
    :param train_loader:
    :param eval_loader:
    :param train_params:
    :param logger:
    :return:
    """
    metrics = train_utils.get_zeroed_metrics_dict()
    best_eval_score = 0

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=train_params.lr, momentum=0.1, nesterov=True)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_params.lr_step_size,
                                                gamma=train_params.lr_gamma)

    x_ref, y_ref, = next(iter(train_loader))

    for epoch in tqdm(range(train_params.num_epochs)):
        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()

        for i, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            y_hat = model(x)

            # img = torch.squeeze(x).cpu().detach().numpy()
            # plt.imshow(img, cmap='gray')
            # plt.show()
            #
            # img = torch.squeeze(y).cpu().detach().numpy()
            # plt.imshow(img, cmap='gray')
            # plt.show()

            # img = torch.squeeze(y_hat).cpu().detach().numpy()
            # plt.imshow(img, cmap='gray')
            # plt.show()

            y_hat = y_hat.reshape(-1)
            y = y.reshape(-1)
            # loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
            # my_loss = eval(train_params.loss)

            my_loss = train_utils.get_loss(train_params.loss)

            loss = my_loss(y_hat, y)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params.grad_clip)
            metrics['count_norm'] += 1

            # NOTE! This function compute scores correctly only for one hot encoding representation of the logits
            # batch_score = train_utils.compute_score_with_logits(y_hat, y.data).sum()
            # metrics['train_score'] += batch_score.item()

            metrics['train_loss'] += loss.item()

            # Report model to tensorboard
            if epoch == 0 and i == 0:
                logger.report_graph(model, x)

        # x_img = torch.squeeze(x).cpu().detach().numpy()
        # y_img = torch.squeeze(y).cpu().detach().numpy()
        # y_hat_img = torch.squeeze(y_hat).cpu().detach().numpy()
        # for i in range(len(x_img)):
        #     wandb.log({"input": wandb.Image(x_img[i]),
        #                "reference": wandb.Image(y_img[i]),
        #                "output": wandb.Image(y_hat_img[i])})
        x_ref, y_ref, = next(iter(train_loader))
        out = model(x_ref)

        x_ref = np.squeeze(x_ref[0].cpu().detach().numpy())
        y_ref = np.squeeze(y_ref[0].cpu().detach().numpy())
        out = np.squeeze(out[0].cpu().detach().numpy())

        x_ref *= 255 / np.max(x_ref)
        y_ref *= 255 / np.max(y_ref)
        out *= 255 / np.max(out)


        wandb.log({"input": wandb.Image(x_ref),
                   "reference": wandb.Image(y_ref),
                   "output": wandb.Image(out)})

        # Learning rate scheduler step
        scheduler.step()

        # Calculate metrics
        metrics['train_loss'] /= len(train_loader.dataset)

        metrics['train_score'] /= len(train_loader.dataset)
        metrics['train_score'] *= 100

        norm = metrics['total_norm'] / metrics['count_norm']

        model.train(False)
        metrics['eval_score'], metrics['eval_loss'] = evaluate(model, eval_loader, train_params)
        model.train(True)

        epoch_time = time.time() - t
        logger.write_epoch_statistics(epoch, epoch_time, metrics['train_loss'], norm,
                                      metrics['train_score'], metrics['eval_score'])

        scalars = {'Accuracy/Train': metrics['train_score'],
                   'Accuracy/Validation': metrics['eval_score'],
                   'Loss/Train': metrics['train_loss'],
                   'Loss/Validation': metrics['eval_loss']}
        wandb.log(scalars)
        logger.report_scalars(scalars, epoch)

        if metrics['eval_score'] > best_eval_score:
            best_eval_score = metrics['eval_score']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer)

    return get_metrics(best_eval_score, metrics['eval_score'], metrics['train_loss'])


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, train_params: TrainParams) -> Scores:
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    score = 0
    loss = 0

    for i, (x, y) in tqdm(enumerate(dataloader)):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        y_hat = model(x)
        y_hat = y_hat.reshape(-1)
        y = y.reshape(-1)

        my_loss = train_utils.get_loss(train_params.loss)
        loss += my_loss(y_hat, y)
        # score += train_utils.compute_score_with_logits(y_hat, y).sum().item()
    if train_params.exp_type == 'wall':

        for i, y_hat_i in enumerate(y_hat):
            if i == len(y_hat) - 1:
                if abs(y_hat_i - y[i]) < abs((y[i - 1] + y[0]) / 2 - y[i]):
                    score = score + 1
            else:
                if abs(y_hat_i - y[i]) < abs((y[i - 1] + y[i + 1]) / 2 - y[i]):
                    score = score + 1

    else:
        if torch.norm(y_hat - y, 2) < torch.tensor(0.01):
            score = score + 1

    loss /= len(dataloader.dataset)
    score /= len(dataloader.dataset)
    score *= 100
    return score, loss
