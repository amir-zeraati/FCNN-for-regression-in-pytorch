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

# modify it to automatic later
# def interpret(model, X, y, predicted, output_dir, cell_line, marks,
#               baseline_method='zero'):
#     ig = IntegratedGradients(model)
#     rows = []
#     if baseline_method == 'zero':
#         baseline = torch.zeros(1, 11, requires_grad=True).to(device)
#     if baseline_method == 'mean':
#         baseline = torch.mean(X, dim=0).requires_grad_()
#         baseline = torch.reshape(baseline, (1, 11))
#     if baseline_method == 'zero_output':
#         baseline = torch.Tensor(np.array([1, 0.75, 0.8, 0.9, 1, 0.75, 1,
#                                           1.5, 0.75, 1, 1]).reshape(1,
#                                                                     11)
#                                 ).requires_grad_().to(device)
#     for i, input in enumerate(X):
#         input.reshape(1, 11)
#         attributions = ig.attribute(
#             input, baseline, return_convergence_delta=False)
#         attributions = attributions.detach().cpu()
#         attributions = np.array(attributions)
#         rows = np.append(rows, attributions)
#         rows = np.append(rows, y[i])
#         rows = np.append(rows, predicted[i])
#     rows = np.array(rows)
#     rows = np.reshape(rows, (-1, 13), order='C')
#     columns_attr = ['Attribution_H2A.Z', 'Attributions_H3K27ac',
#                     'Attributions_H3K79me2', 'Attributions_H3K27me3',
#                     'Attributions_H3K9ac', 'Attributions_H3K4me2',
#                     'Attributions_H3K4me3', 'Attributions_H3K9me3',
#                     'Attributions_H3K4me1', 'Attributions_H3K36me3',
#                     'Attributions_H4K20me1', 'PODLS', "Predicted_PODLS"]
#     attributions = pd.DataFrame(rows,
#                                 columns=columns_attr)
#     # if the model is log to raw
#     X = 10**X.cpu()
#     df = pd.DataFrame(X.cpu(), columns=marks)
#     attributions = pd.concat([attributions, df], axis=1).to_csv(
#         '{}{}_{}_attributions.csv'.format(output_dir,
#                                           cell_line,
#                                           baseline_method))
#     print('IG Attributions:', attributions)
#     # print('Convergence Delta:', delta)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default='FCNN')
    parser.add_argument('--preprocessing', type=str, default='min max normalization')
    parser.add_argument('--max_epoch', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--listfile', nargs='+', type=str,
                        default='data/K562_2000_merged_histones_init.csv.gz')
    parser.add_argument('--marks', nargs='+', type=str,
                        default=['H2A.Z', 'H3K27ac', 'H3K79me2', 'H3K27me3',
                                 'H3K9ac', 'H3K4me2', 'H3K4me3', 'H3K9me3',
                                 'H3K4me1', 'H3K36me3', 'H4K20me1'])
    parser.add_argument('--output', type=str, default=['initiation'])
    parser.add_argument('--output_dir', type=str,
                        default='development/')
    parser.add_argument('--image_format', type=str, default='png')

    args = parser.parse_args()

    df = pd.read_csv('{}'.format(args.listfile), compression='gzip')
    masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
    print('Number of NANs is {}'.format(masks['signal'].sum()))
    df.loc[~masks['signal'].astype(bool)] = np.nan
    df = df.dropna()
    min_init = np.min(df['initiation'])
    max_init = np.max(df['initiation'])
    print(df)
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
            # costumize it to your case, try to make it automatic
            for i in args.marks + ['initiation']:
                df[i] = (df[i] - np.min(df[i])) / (
                    np.max(df[i]) - np.min(df[i]))
        
        # costumize it to your case, try to make it automatic
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        X_train, y_train = shuffle(X_train, y_train)
        X_val = torch.tensor(X_train[0:100000], dtype=float32).to(device)
        y_val = torch.tensor(y_train[0:100000], dtype=float32).to(device)
        X_train = torch.tensor(X_train[100000:], dtype=float32).to(device)
        X_test = torch.tensor(X_test, dtype=float32).to(device)
        y_train = torch.tensor(y_train[100000:], dtype=float32).to(device)
        X_train = X_train.transpose(1, 2).contiguous()
        X_test = X_test.transpose(1, 2).contiguous()
        X_val = X_val.transpose(1, 2).contiguous()
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
        # input = torch.rand(1, 11).to(device)
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
        X = df[args.marks].to_numpy()
        X = torch.tensor(X, dtype=float32).to(device)
        predicted_test = model(X_test).cpu().detach().numpy()
        predicted = model(X_train).cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        # p1 = max(max(predicted), max(y_train))
        # p2 = min(min(predicted), min(y_train))
        p1 = -2
        p2 = 2
        y = model(X).cpu().detach().numpy()
        if args.preprocessing == 'min max normalization':
            y = y * (np.max(df['initiation']) - np.min(
                                       df['initiation'])) + np.min(
                                       df['initiation'])
            df['initiation'] = df['initiation'] * (np.max(
                               df['initiation']) - np.min(
                               df['initiation'])) + np.min(df['initiation'])
        df['predicted'] = y
        plt.plot([p1, p2], [p1, p2], '-', color='orange')
        report(predicted, predicted_test, y_train, y_test, min_init, max_init,
               preprocessing=args.preprocessing, output_dir=args.output_dir,
               image_format=args.image_format, cell_line=args.cell_line)

    if args.method == 'Integrated gradients':
        df = pd.read_csv('{}'.format(args.listfile), compression='gzip')
        masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
        print('Number of NANs is {}'.format(masks['signal'].sum()))
        df.loc[~masks['signal'].astype(bool)] = np.nan
        df = df.dropna()
        if args.preprocessing == 'log to raw':
            for i in args.marks:
                df[i] = df[i] + np.min(df[i][(df[i] != 0)])
                df[i] = np.log10(df[i])
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', 'initiation'].to_numpy()
        X_test = torch.tensor(X_test, dtype=float32).to(device)
        model = MLP()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load('development/model_weights.pth'))
        model.to(device)
        model.eval()
        predicted = model(X_test).detach().cpu().numpy()
        interpret(model, X_test, y_test, predicted, output_dir=args.output_dir,
                  cell_line=args.cell_line, marks=args.marks,
                  baseline_method='zero')
    # add it to FCNN, try if you can generaliza it to apply for future models
    if args.method == 'log FCNN Gridsearch':
        for i in args.marks:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test = torch.tensor(X_test, dtype=float32)
        y_test = torch.tensor(y_test, dtype=float32)
        X_val = torch.tensor(X_train[0:100000], dtype=float32)
        y_val = torch.tensor(y_train[0:100000], dtype=float32)
        X_train = torch.tensor(X_train[100000:], dtype=float32)
        y_train = torch.tensor(y_train[100000:], dtype=float32)
        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size)
        valid_ds = TensorDataset(X_val, y_val)
        valid_dl = DataLoader(valid_ds, batch_size=args.batch_size)

        def train_mlp(config, checkpoint_dir='development/', data_dir=None):
            # net = MLP(config["units"], config["l1"], config["l2"])
            # net = MLP(config["units"])
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
            # optimizer = optim.Adam()

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