import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

transform = transforms.ToTensor() # 画像をテンソルへ変換するtransformモジュール

# MNISTデータセットを読み込む
dataset = torchvision.datasets.MNIST(
  root='./data', # データセットの保存場所
  train=True, # 学習データを読み込む
  transform=transform, # 前処理の指定
  download=True # ダウンロードするかどうか
)

# データセットから0番目の画像を選択する
image, label = dataset[0]

print('size:', len(dataset)) 
print('type:', type(image))
print('label:', label) 

# 画像を表示する
# plt.imshow(image, cmap='gray')
# plt.show()


import torch
data_loader = torch.utils.data.DataLodaer(
  dataset,
  batch_size=32,
  shuffle=True
)

for x, label in data_loader:
  print('x shape:', x.shape)
  print('label shape:', label.shape)
  break # 0番目のミニバッチの情報のみを表示するため、ここでループを抜ける