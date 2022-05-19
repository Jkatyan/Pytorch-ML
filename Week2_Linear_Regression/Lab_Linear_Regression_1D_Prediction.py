"""
Lab 1) Linear Regression for 1D Prediction
"""
import torch
from torch import nn
from torch.nn import Linear

torch.manual_seed(1)  # Linear parameters are randomly generated, so set a manual seed to ensure the same inputs

""""
Notes:
Linear Regression in 1D is an equation of the form y = b + wx,
where b = bias, w = weight (the slope).

Because these values must be trained, the parameter requires_grad must be true.
"""
# Define parameters
weight = torch.tensor(2.0, requires_grad=True)
bias = torch.tensor(1.0, requires_grad=True)

# Make a prediction function
def forward(x, w, b):
    return b + w * x

# Make the prediction at x = 1
print("Predict forward(1): ", forward(1, weight, bias))

# Make multiple input prediction
input = torch.tensor([[1.0], [2.0]])
print("Predict forward([1, 2]): ", forward(input, weight, bias))

# Create a linear regression model using torch.nn
lr = Linear(in_features=1, out_features=1, bias=True)
print("weight:", lr.weight[0].item())
print("bias:", lr.bias[0].item())

# Make a prediction using the linear regression model
x = torch.tensor([1.0])
y = lr(x)
print("Prediction: ", y)

# Building Custom Modules
class LinReg(nn.Module):
    # Constructor
    def __init__(self, input_size, output_size):
        super(LinReg, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size, bias=True)
    # Prediction
    def forward(self, x):
        return self.linear(x)

# Create object and predict
model = LinReg(1, 1)
prediction = model.forward(torch.tensor([1.0]))
print(prediction)

