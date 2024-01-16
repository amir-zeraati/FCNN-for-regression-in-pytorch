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
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

        
# the dataset is from sklearn, you can use your own dataset
data = fetch_california_housing()
print(data.feature_names)
# df = pd.read_csv('{}'.format(args.listfile), compression='gzip')

X, y = data.data, data.target
print(X.shape, y.shape)
# train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                    shuffle=True)
num_inputs = X.shape[1]
num_outputs = 1
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--preprocessing', type=str, default='min max normalization')
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    print(torch.cuda.is_available())
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

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
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)

    # X_train, y_train = shuffle(X_train, y_train)
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size)
    valid_ds = TensorDataset(X_val, y_val)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size)

    def train_mlp(config, checkpoint_dir='development/', data_dir=None):
        net = MLP(config["units"])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
        net.to(device)

        criterion = nn.MSELoss()
        # optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
        # optimizer = optim.Adam(net.parameters(), lr=config["lr"])
        optimizer = optim.Adam(net.parameters())

        if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
        for epoch in range(160):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(train_dl, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                    running_loss /
                                                    epoch_steps))
                    running_loss = 0.0

            # Validation loss
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, data in enumerate(valid_dl, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = criterion(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()),
                            path)

            tune.report(loss=(val_loss / val_steps))
            print("Finished Training")

    def test_loss(net, device="cpu"):
        with torch.no_grad():
            outputs = net(X_test)
            outputs = outputs.cpu().numpy()
        return mean_squared_error(outputs, y_test.cpu().numpy())

    # def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    #     config = {
    #         "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    #         "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    #         "lr": tune.loguniform(1e-4, 1e-1),
    #         "units": tune.qrandint(40, 100),
    #         "batch_size": tune.choice([2, 4, 8, 16])
    #     }
    def main1(num_samples=20, max_num_epochs=160, gpus_per_trial=4):
        config = {
            "units": tune.qrandint(40, 1000),
            "batch_size": tune.choice([32, 64, 128]),
            # "lr": tune.loguniform(1e-4, 1e-1)
        }
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=30,
            reduction_factor=2)
        reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["loss", "accuracy", "training_iteration"])
        result = tune.run(
            train_mlp,
            resources_per_trial={"cpu": 10, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter)

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        # print("Best trial final validation accuracy: {}".format(
        #     best_trial.last_result["accuracy"]))

        # best_trained_model = MLP(best_trial.config["units"] ,best_trial.config["l1"], best_trial.config["l2"])
        best_trained_model = MLP(best_trial.config["units"])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(
            best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)

        test_acc = test_loss(best_trained_model, device)
        print("Best trial test set loss: {}".format(test_acc))

    main1(num_samples=20, max_num_epochs=300, gpus_per_trial=2)