"""
Lab 1) 1D Tensors
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Version
print("Running On: PyTorch v" + str(torch.__version__), end="\n\n")

# Make a tensor
int_tensor = torch.tensor([0, 1, 2, 3, 4, 5])
float_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

# Cast a tensor
int_to_float = int_tensor.type(torch.FloatTensor)

# Resize a tensor
print(int_tensor.size())
print(int_tensor.view(6, 1).size())  # (Turn 1D tensor into 2D)

# Numpy support
numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
numpy_to_tensor = torch.from_numpy(numpy_array)  # To Numpy
back_to_numpy = numpy_to_tensor.numpy()  # To Tensor

# Pandas support
pandas_series = pd.Series([0.1, 2, 0.3, 10.1])
pandas_tensor = torch.from_numpy(pandas_series.values)

# Index value
tensor_a = torch.tensor([0, 1, 2, 3])
print(tensor_a[1].item())

# Set value
tensor_a[1] = 100
print(tensor_a[1].item())

# Tensor subset
tensor_subset = tensor_a[1:4]
print(tensor_subset)

tensor_a[0:3] = torch.tensor([100, 200, 300])
print(tensor_a)

tensor_a[[0, 3]] = 10
print(tensor_a)

# Tensor math
tensor_a = torch.tensor([1.0, 2.0, 3.0])
mean = tensor_a.mean()
standard_deviation = tensor_a.std()
max_val = tensor_a.max()
min_val = tensor_a.min()

pi_tensor = torch.tensor([0, np.pi/2, np.pi])
sin = torch.sin(pi_tensor)

# Torch linspace (generate numbers over an interval)
len_5_tensor = torch.linspace(-2, 2, steps=5, dtype=torch.int64)
print(len_5_tensor)

# PRACTICE PROBLEM
# Use both torch.linspace() and torch.sin() to construct a tensor that contains the 100 sin result in range from 0 (0
# degree) to 2π (360 degree):
circle_tensor = torch.linspace(0, 2*np.pi, 100)
sin_circle_tensor = torch.sin(circle_tensor)
plt.plot(circle_tensor.numpy(), sin_circle_tensor.numpy())
plt.show()
