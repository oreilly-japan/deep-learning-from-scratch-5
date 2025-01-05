# import torch
import torch.nn as nn

# # torch.nn.Moduleを継承したサブクラスをModuleクラスとして定義
# class Model(nn.Module):
#   # 最初にオブジェクトを作るときに呼び出される特殊な関数「コンストラクタ（constructor）」
#   # コンストラクタは、オブジェクトの状態を、最初の状態にするときの処理をするのに使います。
#   def __init__(self):
#     super().__init__()
#     self.W = nn.Parameter(torch.zeros((1, 1)))
#     self.b = nn.Parameter(torch.zeros(1))
  
#   def forward(self, x):
#     y = x @ self.W + self.b
#     return y
  
class Model_v2(nn.Module):
  def __init__(self, input_size=1, output_size=1):
    super().__init__()
    self.linear = nn.Linear(input_size, output_size) # Linearクラスを使用

  def forward(self, x):
    y = self.linear(x) # 以前は y = x @ self.W + self.b
    return y

model = Model_v2()

# モデルにあるすべてのパラメータにアクセスできる
for param in model.parameters():
  print(param)