"""
Lab 2) 2D Tensors
"""
import torch

# Create 2D Tensor
tensor = torch.tensor([[0, 1, 2],
                       [3, 4, 5],
                       [6, 7, 8]])

print("Shape: " + str(tensor.shape))
print("# Dimensions: " + str(tensor.ndimension()))

# Retrieving values
val_a = tensor[0][0].item()
val_b = tensor[1][0].item()
val_c = tensor[0][1].item()
print(val_a)
print(val_b)
print(val_c)
