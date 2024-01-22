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
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from torch import float32, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# the dataset is from sklearn, you can use your own dataset
data = fetch_california_housing()
print(data.feature_names)
# df = pd.read_csv('{}'.format(args.listfile), compression='gzip')

X, y = data.data, data.target
print(X.shape, y.shape)
# train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                    shuffle=True)

torch.cuda.empty_cache()
loss_func = F.mse_loss

# make it automatic Later
num_inputs = X.shape[1]
num_outputs = 1
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


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


def fit(epochs, model, loss_func, opt, train_dl, valid_dl=None):
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


def report(predicted, predicted_test, y_train, y_test,
           preprocessing, output_dir, image_format):
    print(mean_squared_error(predicted, y_train))
    print(mean_squared_error(predicted_test, y_test))
    plt.scatter(y_train.ravel(), predicted, s=0.1)
    plt.title('Predicted values with respect to the observed values')
    plt.ylabel('Predicted vlaues')
    plt.xlabel('Observed values')
    plt.axis('square')
    plt.savefig('{}{}distribution_performance.{}'.format(output_dir,
                                                         preprocessing,
                                                         image_format),
                dpi=300, bbox_inches='tight')
    plt.savefig('{}{}distribution_performance.eps'.format(output_dir,
                                                          preprocessing),
                dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(5, 5))
    plt.plot([p1, p2], [p1, p2], 'w-')
    plt.hist2d(y_train.ravel(),
               predicted.ravel(),
               bins=[300, 300], cmap=plt.cm.nipy_spectral)
    plt.colorbar()
    plt.xlabel('observed values')
    plt.ylabel('predicted values')
    plt.title('Predicted values with respect to the observed values')
    plt.savefig('{}{}.{}'.format(output_dir, preprocessing,
                                 image_format),
                dpi=300, bbox_inches='tight', transparent=False)
    plt.savefig('{}{}.eps'.format(output_dir, preprocessing),
                dpi=300, bbox_inches='tight', transparent=False)
    plt.close()
    plt.plot(y_train[0:50], '-o')
    plt.plot(predicted[0:50], '-o')
    plt.legend(['Observed', 'Predicted'], loc='upper right')
    plt.title('Comparison of observed values and predicted values by {}'.format(args.method))
    plt.savefig('{}{}comaprison_r_p.{}'.format(output_dir, preprocessing,
                                               image_format),
                dpi=300, bbox_inches='tight', transparent=False)
    plt.savefig('{}{}comaprison_r_p.eps'.format(output_dir,
                                                preprocessing),
                dpi=300, bbox_inches='tight', transparent=False)
    plt.close()


def interpret(model, X, y, predicted, output_dir,
              baseline_method='zero', num_inputs=num_inputs,
              feature_names=data.feature_names):
    ig = IntegratedGradients(model)
    rows = []
    if baseline_method == 'zero':
        baseline = torch.zeros(1, num_inputs, requires_grad=True).to(device)
    for i, input in enumerate(X):
        input.reshape(1, num_inputs)
        attributions = ig.attribute(
            input, baseline, return_convergence_delta=False)
        attributions = attributions.detach().cpu()
        attributions = np.array(attributions)
        rows = np.append(rows, attributions)
        rows = np.append(rows, y[i])
        rows = np.append(rows, predicted[i])
    rows = np.array(rows)
    rows = np.reshape(rows, (-1, num_inputs + 2), order='C')
    columns_attr = ["attr_" + feature for feature in feature_names]
    columns_attr.append('observed')
    columns_attr.append('predicted')
    attributions = pd.DataFrame(rows,
                                columns=columns_attr)
    df = pd.DataFrame(X.cpu(), columns=feature_names)
    attributions = pd.concat([attributions, df], axis=1)
    attributions.to_csv('{}{}_attributions.csv'.format(output_dir,
                                                       baseline_method))
    print('IG Attributions:', attributions)
    # print('Convergence Delta:', delta)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default='FCNN')
    parser.add_argument('--preprocessing', type=str, default='min max normalization')
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_dir', type=str,
                        default='development/')
    parser.add_argument('--image_format', type=str, default='png')

    args = parser.parse_args()

    print(torch.cuda.is_available())
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    writer = SummaryWriter(args.output_dir)
    if args.method == 'FCNN':
        model, opt = mlp()
        model = nn.DataParallel(model)
        model = model.to(device)
        if args.preprocessing == 'min max normalization':
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            scaler_y = MinMaxScaler()
            scaler_y.fit(y_train)
            y_train = scaler_y.transform(y_train)
            y_test = scaler_y.transform(y_test)
        # validation split
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=0.25,
                                                          shuffle=True)
        # 0.25 x 0.8 = 0.2
        X_train = torch.tensor(X_train,
                               dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train,
                               dtype=torch.float32).reshape(-1, 1).to(device)
        X_test = torch.tensor(X_test,
                              dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test,
                              dtype=torch.float32).reshape(-1, 1).to(device)
        X_val = torch.tensor(X_val,
                             dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val,
                             dtype=torch.float32).reshape(-1, 1).to(device)

        # X_train, y_train = shuffle(X_train, y_train)
        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size)
        valid_ds = TensorDataset(X_val, y_val)
        valid_dl = DataLoader(valid_ds, batch_size=args.batch_size)

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in opt.state_dict():
            print(var_name, "\t", opt.state_dict()[var_name])
        model.eval()
        torch.manual_seed(123)
        np.random.seed(123)
        model, val_loss_list, train_loss_list = fit(args.max_epoch, model,
                                                    loss_func, opt, train_dl,
                                                    valid_dl)
        torch.save(model.state_dict(), '{}{}'.format(
            args.output_dir, 'model_weights.pth'))
        plt.plot(train_loss_list)
        plt.plot(val_loss_list)
        plt.title('Loss during training')
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Epochs')
        plt.scatter(np.argmin(val_loss_list),
                    np.min(val_loss_list), facecolors='none',
                    edgecolors='chocolate', s=50)
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig('{}loss.{}'.format(args.output_dir, args.image_format),
                    dpi=300, bbox_inches='tight')
        plt.close()
        with torch.no_grad():
            predicted_test = model(X_test).cpu().detach().numpy()
            predicted = model(X_train).cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()
        p1 = max(max(predicted), max(y_train))
        p2 = min(min(predicted), min(y_train))
        plt.plot([p1, p2], [p1, p2], '-', color='orange')
        report(predicted, predicted_test, y_train, y_test,
               preprocessing=args.preprocessing, output_dir=args.output_dir,
               image_format=args.image_format)
        # model = MLP()
        # model = nn.DataParallel(model)
        # model.load_state_dict(torch.load('development/model_weights.pth'))
        # model.to(device)
        # model.eval()
        interpret(model, X_test, y_test, predicted_test,
                  output_dir=args.output_dir,
                  baseline_method='zero', num_inputs=num_inputs)
