import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.rand(100, 1)
y = torch.sin(2 * torch.pi * x) + torch.rand(100, 1)

# a = torch.rand(100, 5) # ランダムなテンソル

# b = F.sigmoid(a) # シグモイド関数
# c = F.relu(a) # ReLu関数


class Model(nn.Module):
  def __init__(self, input_size=1, hidden_size=10, output_size=1):
    super().__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    y = self.linear1(x)
    y = F.sigmoid(y)
    y = self.linear2(y)
    return y
  
lr = 0.2
iters = 10000

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for i in range(iters):
  y_pred = model(x)
  loss = F.mse_loss(y_pred, y)

  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  if i % 1000 == 0:
    print("Step {}: loss = {}".format(i, loss.item()))
print(loss.item())