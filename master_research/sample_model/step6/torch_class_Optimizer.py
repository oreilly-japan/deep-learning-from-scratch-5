# 最適化手法は、ネットワークのパラメータを更新するためのアルゴリズムです。
# オプティマイザは、その最適化手法を具体的に実装したものです。
# torch_class_Linear.pyはオプティマイザを使うと次のようになります。

import torch
import torch.nn as nn

torch.manual_seed(0)
# トイデータセット
x = torch.rand(100, 1)
y = 2 * x + 5 + torch.rand(100, 1)

lr = 0.1
iters = 100

class Model(nn.Module):
  # 最初にオブジェクトを作るときに呼び出される特殊な関数「コンストラクタ（constructor）」
  # コンストラクタは、オブジェクトの状態を、最初の状態にするときの処理をするのに使います。
  def __init__(self):
    super().__init__()
    self.W = nn.Parameter(torch.zeros((1, 1)))
    self.b = nn.Parameter(torch.zeros(1))
  
  def forward(self, x):
    y = x @ self.W + self.b
    return y

model = Model()
# SGDなりAdamなりのオプティマイザが使える
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # オプティマイザの生成

for i in range(iters):
  y_hat = model(x)
  loss = nn.functional.mse_loss(y_hat, y)

  loss.backward()
  optimizer.step() # パラメータの更新
  optimizer.zero_grad() # 勾配のリセット

  if i % 10 == 0: # 10回ごとに出力
    print("Step {}: loss = {}".format(i, loss.item()))
print('====')
print(loss.item())
print('W =', model.W.item())
print('b =', model.b.item())