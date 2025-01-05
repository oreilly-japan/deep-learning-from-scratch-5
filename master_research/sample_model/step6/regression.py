import torch
import torch.nn.functional as F

# トイデータセット
torch.manual_seed(0)
x = torch.randn(100, 1)
y = 2 * x + 5 + torch.randn(100, 1)

W = torch.zeros((1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def predict(x):
  y = x @ W + b
  return y

def mean_squared_error(x0, x1):
  diff = x0 - x1
  N = len(diff)
  return torch.sum(diff ** 2) / N

lr = 0.1
iters = 100

for i in range(iters):
  y_hat = predict(x)
  # loss = mean_squared_error(y, y_hat)
  loss = F.mse_loss(y_hat, y)

  loss.backward()

  # 値の更新
  W.data -= lr * W.grad.data
  b.data -= lr * b.grad.data

  # 勾配のリセット
  W.grad.zero_()
  b.grad.zero_()

  if i % 10 == 0: # 10回ごとに出力
    print("Step {}: loss = {}".format(i, loss.item()))
print('====')
print(loss.item())
print('W =', W.item())
print('b =', b.item())
