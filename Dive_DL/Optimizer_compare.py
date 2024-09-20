import copy

import os

os.environ["QT_QPA_PLATFORM"] = "wayland"

import torch
from torchvision import datasets
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.optim import SGD, Adam, Adagrad, RMSprop
from metrics import accuracy
from model import get_model
from plots import multiple_plot, single_plot


import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from data import get_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'





def train_batch(x, y, model, loss_fn, optimizer):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()




@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()


def train_with_optimizer(trn_dl, val_dl, model, loss_fn, _optimizer):
    optimizer = _optimizer
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    for epoch in range(200):
        print(epoch)
        train_epoch_losses, train_epoch_accuracies = [], []
        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            batch_loss = train_batch(x, y, model, loss_fn, _optimizer)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()

        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        for ix, batch in enumerate(iter(val_dl)):
            x, y = batch
            val_is_correct = accuracy(x, y, model)
            validation_loss = val_loss(x, y, model, loss_fn)
        val_epoch_accuracy = np.mean(val_is_correct)

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(validation_loss)
        val_accuracies.append(val_epoch_accuracy)

    epochs = np.arange(200) + 1
    # epochs = np.arange(2) + 1
    return epochs, train_losses, val_losses, train_accuracies, val_accuracies


def multiple_train():
    trn_dl, val_dl = get_data(device)
    SGD_model, SGD_loss_fn, = get_model(active_name='relu', device=device)
    optimizer_sgd = SGD(SGD_model.parameters(), lr=0.001, weight_decay=0.1)
    Adam_model, Adam_loss_fn = get_model(active_name='relu', device=device)
    optimizer_adam = Adam(Adam_model.parameters(), lr=0.001)
    Adagrad_model, Adagrad_loss_fn = get_model(active_name='relu', device=device)
    optimizer_Adagrad = Adagrad(Adagrad_model.parameters(), lr=0.001)
    RMSprop_model, RMSprop_loss_fn = get_model(active_name='relu', device=device)
    optimizer_RMSprop = RMSprop(RMSprop_model.parameters(), lr=0.001)
    models = [SGD_model, Adagrad_model, RMSprop_model, Adam_model]
    losses = [SGD_loss_fn, Adagrad_loss_fn, RMSprop_loss_fn, Adam_loss_fn]
    optimizers = [optimizer_sgd, optimizer_Adagrad, optimizer_RMSprop, optimizer_adam]
    epochs, train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], [], []
    for model, loss_fn, optimizer in zip(models, losses, optimizers):
        _epochs, _train_losses, _val_losses, _train_accuracies, _val_accuracies = \
            train_with_optimizer(copy.deepcopy(trn_dl), val_dl, model, loss_fn, optimizer)
        epochs.append(_epochs)
        train_losses.append(_train_losses)
        val_losses.append(_val_losses)
        train_accuracies.append(_train_accuracies)
        val_accuracies.append(_val_accuracies)
    multiple_plot(epochs, train_losses)
    return epochs, train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == '__main__':
    epochs, train_losses, val_losses, train_accuracies, val_accuracies = multiple_train()
