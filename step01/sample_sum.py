import numpy as np
import matplotlib.pyplot as plt

x_means = []
N = 5

for _ in range(10000):
    xs = []
    for i in range(N):
        x = np.random.rand()
        xs.append(x)
    mean = np.sum(xs)
    x_means.append(mean)

# normal distribution
def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y
x_norm = np.linspace(-5, 5, 1000)
mu = 0.5 * N
sigma = np.sqrt(1 / 12 * N)
y_norm = normal(x_norm, mu, sigma)

# plot
plt.hist(x_means, bins='auto', density=True)
plt.plot(x_norm, y_norm)
plt.title(f'N={N}')
plt.xlim(-1, 6)
plt.show()