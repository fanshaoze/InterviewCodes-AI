import copy

import os

os.environ["QT_QPA_PLATFORM"] = "wayland"

import torch
from torchvision import datasets
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.optim import SGD, Adam, Adagrad, RMSprop

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn


def get_activation(name):
    if name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'Hardswish':
        return nn.Hardswish()
    else:
        raise ValueError(f'Activation {name} is not recognized')


def get_model(active_name='relu', device='cpu'):
    active_function = get_activation(active_name)

    model = nn.Sequential(
        nn.Linear(28 * 28, 1000),
        active_function,
        nn.Linear(1000, 10)
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    return model, loss_fn


def get_MLP_model(active_name='relu', device='cpu', dropout=0.5, hidden_layers=4):
    active_function = get_activation(active_name)
    model_list = [nn.Linear(28 * 28, 1000), active_function]
    if dropout:
        model_list.append(nn.Dropout(dropout))
    for i in range(hidden_layers):
        model_list.append(nn.Linear(1000, 1000))
        model_list.append(active_function)
        if dropout:
            model_list.append(nn.Dropout(dropout))
    model_list.append(nn.Linear(1000, 10))
    model = nn.Sequential(*model_list).to(device)

    loss_fn = nn.CrossEntropyLoss()
    return model, loss_fn
