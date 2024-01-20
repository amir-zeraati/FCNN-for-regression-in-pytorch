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
model = MLP()


if __name__ == '__main__':
    with torch.no_grad():
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load('development/model_weights.pth', map_location=device))
    
        if isinstance(model, torch.nn.DataParallel):  # extract the module from dataparallel models
            model = model.module
        model.cpu()
        model.eval()  
        torch_input = torch.randn(1, num_inputs)
        onnx_program = torch.onnx.export(model,
                        torch_input,
                        'development/model.onnx')
    # import onnx
    # onnx_model = onnx.load("development/model.onnx")
    # onnx.checker.check_model(onnx_model)