# a regression task in pytorch

import argparse
import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from torch import float32, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

torch.cuda.empty_cache()
loss_func = F.mse_loss

# make it automatic Later
num_inputs = 11
num_outputs = 1

class MLP(nn.Module):
    def __init__(self, units=100):
        super().__init__()
        self.lin1 = nn.Linear(num_inputs, units)
        self.lin2 = nn.Linear(units, num_outputs)

    def forward(self, xb):
        xb = F.relu(self.lin1(xb))
        xb = F.relu(self.lin2(xb))
        return xb
    
    

def mlp(model=MLP()):
    return model, optim.Adam(model.parameters())


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    # Early stopping
    last_loss = np.inf
    patience = 3
    trigger_times = 0
    val_loss_list = []
    train_loss_list = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            train_losses, aaa = loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
            train_losses, train_nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in train_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_loss_list.append(val_loss)
        train_loss = np.sum(np.multiply(train_losses,
                                        train_nums)) / np.sum(train_nums)
        train_loss_list.append(train_loss)
        print(epoch, val_loss)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        # Early stopping
        current_loss = val_loss
        print('The Current Loss:', current_loss)
        if val_loss <= min(val_loss_list):
            best_model = copy.deepcopy(model)
        if current_loss > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return best_model, val_loss_list, train_loss_list
        else:
            print('trigger times: 0')
            trigger_times = 0
        if epoch + 1 >= epochs:
            return best_model, val_loss_list, train_loss_list
        last_loss = current_loss