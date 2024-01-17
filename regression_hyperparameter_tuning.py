import argparse
import os
import tempfile

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray import train, tune
from ray.air import session
from ray.train import Checkpoint
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import float32, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

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

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

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
train_dl = DataLoader(train_ds, batch_size=128)
valid_ds = TensorDataset(X_val, y_val)
valid_dl = DataLoader(valid_ds, batch_size=128)
test_ds = TensorDataset(X_test, y_test)
test_dl = DataLoader(test_ds, batch_size=128)

class MLP(nn.Module):
    def __init__(self, l1=120, l2=84, units=100):
        super().__init__()
        self.lin1 = nn.Linear(num_inputs, units)
        self.lin2 = nn.Linear(units, num_outputs)

    def forward(self, xb):
        xb = F.relu(self.lin1(xb))
        xb = F.relu(self.lin2(xb))
        return xb


def mlp(model=MLP()):
    return model, optim.Adam(model.parameters())

def train_mlp(config):
    net = MLP(config["l1"], config["l2"], config["units"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    trainloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    valloader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
    )

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):
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
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
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

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")

def test_best_model(best_result):
    best_trained_model = MLP(best_result.config["l1"], best_result.config["l2"], best_result.config["units"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)
    
    testloader = torch.utils.data.DataLoader(
        test_ds, shuffle=False, num_workers=0
    )
    
    with torch.no_grad():
        outputs = best_trained_model(X_test)
        outputs = outputs.cpu().numpy()
        test_loss = mean_squared_error(outputs, y_test.cpu().numpy())

    print("Best trial test set loss: {}".format(test_loss))


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "units": tune.qrandint(40, 1000)
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_mlp),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    
    test_best_model(best_result)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--preprocessing', type=str, default='min max normalization')
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    print(torch.cuda.is_available())
    
    main(num_samples=2, max_num_epochs=2, gpus_per_trial=1)