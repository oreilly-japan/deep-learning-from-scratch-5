import torch

x = torch.tensor(5.0, requires_grad=True)
y = 3 * x ** 2

# dy/dx = 6x
y.backward()
print(x.grad)