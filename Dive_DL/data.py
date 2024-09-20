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


class FMNISTDataset(Dataset):
    def __init__(self, x, y, device):
        x = x.float() / 255
        x = x.view(-1, 28 * 28)
        self.x, self.y, = x, y
        self.device = device

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x.to(self.device), y.to(self.device)

    def __len__(self):
        return len(self.x)


def get_data(device):
    data_folder = '~/data/FMNIST'
    tr_fmnist = datasets.FashionMNIST(data_folder,
                                      download=True,
                                      train=True)
    val_fmnist = datasets.FashionMNIST(data_folder,
                                       download=True,
                                       train=False)
    tr_images, tr_targets = tr_fmnist.data, tr_fmnist.targets
    val_images, val_targets = val_fmnist.data, val_fmnist.targets

    train = FMNISTDataset(tr_images, tr_targets, device)
    val = FMNISTDataset(val_images, val_targets, device)
    tr_dl = DataLoader(train, batch_size=int(len(tr_images) / 50),  # len/50=1200
                       shuffle=True)
    val_dl = DataLoader(val, batch_size=len(val_images),
                        shuffle=False)
    return tr_dl, val_dl
