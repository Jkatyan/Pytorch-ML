"""
Lab 3) 2D Linear Regression in PyTorch
"""
import torch

"""
Start with generating values from -3 to 3 that create a line with a slope of 1 and a bias of -1.
This is the line that you need to estimate.
"""
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 1

"""
Add noise to the data
"""
Y = f + 0.1 * torch.randn(X.size())

"""
Create Model (prediction and loss functions)
"""
weight = torch.tensor(1.0, requires_grad=True)
bias = torch.tensor(2.0, requires_grad=True)

def prediction(x):
    return bias + weight * x

def loss(estimate, actual):
    return torch.mean((estimate - actual) ** 2)

"""
Train model (learning rate / train function)
"""
lr = 0.1

def train_model(i):
    # Loop
    for epoch in range(i):
        # Make prediction
        guess = prediction(X)

        # Calculate Loss
        curr_loss = loss(guess, Y)

        # Compute gradient
        curr_loss.backward()

        # Update weight and bias
        weight.data -= lr * weight.grad.data
        bias.data -= lr * bias.grad.data

        # Zero gradients before running backward in next iteration
        weight.grad.data.zero_()
        bias.grad.data.zero_()

train_model(15)

print("Prediction at x = 5: ", prediction(torch.tensor(5)))