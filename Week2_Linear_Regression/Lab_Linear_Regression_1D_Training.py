"""
Lab 2) Training one parameter for linear regression
"""
import torch
import numpy
import matplotlib.pyplot as plt

class PlotDiagram:
    # Constructor
    def __init__(self, X, Y, w, stop, go=False):
        start = w.data
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [cost(forward(X), Y) for w.data in self.parameter_values]
        w.data = start

    # Executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y, 'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        plt.plot(self.parameter_values.numpy(), self.Loss_function)
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()

    # Destructor
    def __del__(self):
        plt.close('all')

"""
Start by generating a line with a slope of -3 from values -3 to 3
"""
X = torch.linspace(-3, 3, steps=60)
f = -3 * X

"""
Add some noise to simulate real data
"""
Y = f + 1 * torch.randn(X.size())

"""
Create the model / cost function
"""
w = torch.tensor(-10.0, requires_grad=True)

def forward(x):
    return w * x

def cost(prediction, y):  # Mean Squared Error, squares difference between predicted and actual
    return torch.mean((prediction - y) ** 2)

"""
Define learning rate / loss
"""
lr = 0.1
LOSS = []

"""
Train the model
"""
gradient_plot = PlotDiagram(X, Y, w, stop=5)

def train_model(iter):
    for epoch in range(iter):
        # Start by making a prediction
        prediction = forward(X)

        # Calculate the loss
        loss = cost(prediction, Y)

        # Plot diagram
        gradient_plot(prediction, w, loss.item(), epoch)

        # store the loss into list
        LOSS.append(loss.item())

        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()

        # update parameters
        w.data = w.data - lr * w.grad.data

        # zero the gradients before running the backward pass
        w.grad.data.zero_()

# Train Model
train_model(4)

# Plot Loss
plt.plot(LOSS)
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")