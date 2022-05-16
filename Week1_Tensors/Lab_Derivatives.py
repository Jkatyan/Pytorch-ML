"""
Lab 2) 2D Tensors
"""
import torch
import matplotlib.pylab as plt

# Calculate the derivative of a function at a value x
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward()
print("Derivative of y = x^2 + 2x + 1 at x = 2: " + str(x.grad))

# Function to calculate derivative at a point for a function
def derivative_at_point(function, tensor):
    function.backward()
    return tensor.grad

tensor = torch.tensor(2.0, requires_grad=True)
function = tensor ** 2 + 2 * tensor + 1
print(derivative_at_point(function, tensor))

# Partial Derivatives Example
u = torch.tensor(2.0, requires_grad=True)
v = torch.tensor(3.0, requires_grad=True)
function = u * v**2 + v * u**2
function.backward()
print("Partial derivative of u for uv^2 + vu^2: " + str(u.grad))
print("Partial derivative of v for uv^2 + vu^2: " + str(v.grad))

# Scalar valued function for multiple values
x = torch.linspace(0, 10, 10, requires_grad=True)
Y = x**2
y = torch.sum(x**2)  # Create a scalar valued function
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()

# Take the derivative of Relu with respect to multiple value. Plot out the function and its derivative
x = torch.linspace(-10, 10, 1000, requires_grad = True)
Y = torch.relu(x)
y = Y.sum()
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()