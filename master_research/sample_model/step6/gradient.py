import torch

# ローゼンブロック関数の
# (x0, x1) = (0, 2)における微分、つまりdy/dx0, dy/dx1を求める

def rosenbrock(x0, x1):
  y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
  return y

# # x0 = 0, x1 = 2での微分を求める
# # 勾配を計算するためには、requires_grad=Trueを指定する
# # dy/dx0 = -400x0(x1 - x0^2) - 2(1 - x0)
# # dy/dx1 = 200(x1 - x0^2)
# # なので、(x0, x1) = (0, 2)のとき、dy/dx0 = -2, dy/dx1 = 400
# x0 = torch.tensor(0.0, requires_grad=True)
# x1 = torch.tensor(2.0, requires_grad=True)

# y = rosenbrock(x0, x1)
# y.backward()
# print(x0.grad, x1.grad)


x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)

lr = 0.001 # 学習率
iters = 10000 # 繰り返し回数(学習回数)

for i in range(iters):
  if i % 1000 == 0:
    print("Step {}: x0 = {}, x1 = {}".format(i, x0.item(), x1.item()))
  
  y = rosenbrock(x0, x1)
  y.backward()

  # 値の更新
  x0.data -= lr * x0.grad.data
  x1.data -= lr * x1.grad.data

  # 勾配のリセット
  x0.grad.data.zero_()
  x1.grad.data.zero_()
print("Step {}: x0 = {}, x1 = {}".format(iters, x0.item(), x1.item()))