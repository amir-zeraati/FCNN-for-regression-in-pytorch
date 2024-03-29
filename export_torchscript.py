import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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


class MLP(nn.Module):
    def __init__(self, units=100):
        super().__init__()
        self.lin1 = nn.Linear(num_inputs, units)
        self.lin2 = nn.Linear(units, num_outputs)

    def forward(self, xb):
        xb = F.relu(self.lin1(xb))
        xb = F.relu(self.lin2(xb))
        return xb


print(torch.cuda.is_available())
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
device = torch.device("cpu")
model = MLP()


if __name__ == '__main__':
    with torch.no_grad():
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load('development/model_weights.pth',
                                         map_location=device))
        model.eval()
    if isinstance(model, torch.nn.DataParallel):  # extract the module from dataparallel models
        model = model.module
    # Adding dynamic quantization to the model, if you don't want to quantize the model, comment the following lines
    model = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
    # save the model
    torch.save(model.state_dict(), 'development/quantized_model_weights.pth')
    scripted_model = torch.jit.script(model)
    
    print(scripted_model.code)
    scripted_model.save('development/scripted_model.pt')
