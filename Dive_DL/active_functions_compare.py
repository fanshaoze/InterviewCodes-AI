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

def train_with_optimizer(trn_dl, val_dl, model, loss_fn, _optimizer, ep):
    optimizer = _optimizer
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    for epoch in range(ep):
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

    epochs = np.arange(ep) + 1
    # epochs = np.arange(2) + 1
    return epochs, train_losses, val_losses, train_accuracies, val_accuracies

def multiple_activation_train():
    trn_dl, val_dl = get_data(device)
    activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'gelu', 'elu', 'Hardswish']
    epochs, train_losses = [], []
    for activation in activations:
        model, loss_fn = get_model(activation, device)
        optimizer_adam = Adam(model.parameters(), lr=0.001)
        epoch, train_loss, _, _, _ = \
            train_with_optimizer(trn_dl, val_dl, model, loss_fn, optimizer_adam, ep=200)
        epochs.append(epoch)
        train_losses.append(train_loss)
    multiple_plot(epochs, train_losses, 'activation_compare.png', activations)
    return epochs, train_losses


if __name__ == '__main__':
    epochs, train_losses = multiple_activation_train()
    print(train_losses)
