import numpy as np

def kl(p, q):
    return p[0] * np.log(p[0] / q[0]) + p[1] * np.log(p[1] / q[1])

p = [0.7, 0.3]
q = [0.6, 0.4]
print(kl(p, q))  # 0.02160085414354654

p = [0.7, 0.3]
q = [0.2, 0.8]
print(kl(p, q))  # 0.5826853020432394

p = [0.7, 0.3]
q = [0.7, 0.3]
print(kl(p, q))  # 0.0
