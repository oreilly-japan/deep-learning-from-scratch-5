import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms


x = torch.randn(3, 64, 64)
T = 1000
betas = torch.linspace(0.0001, 0.02, T)

for t in range(T):
    beta = betas[t]
    eps = torch.randn_like(x)
    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps

# load image
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'flower.png')
image = plt.imread(file_path)
print(image.shape)  # (64, 64, 3)

# preprocess
preprocess = transforms.ToTensor()
x = preprocess(image)
print(x.shape)  # (3, 64, 64)

def reverse_to_img(x):
    x = x * 255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)

T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)

for t in range(T):
    if t % 100 == 0:
        img = reverse_to_img(x)
        #plt.imshow(img)
        img.save('flower_diffusion_{:04d}.png'.format(t))
        plt.show()

    beta = betas[t]
    eps = torch.randn_like(x)
    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps

# ============================================
# q(x_t|x_{t+1})
# ============================================
def add_noise(x_0, t, betas):
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bar = alpha_bars[t]

    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps
    return x_t

t = 100
x_t = add_noise(x, t, betas)