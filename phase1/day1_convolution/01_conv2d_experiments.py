import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


def conv2d_naive(x, kernel, stride=1, padding=0):

    H, W = x.shape
    KH, KW = kernel.shape

    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding)))

    H_p, W_p = x.shape
    out_h = (H_p - KH) // stride + 1
    out_w = (W_p - KW) // stride + 1

    out = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = x[i:i+KH, j:j+KW]
            out[i, j] = (patch * kernel).sum()

    return out

x = np.array([
    [1, 2, 0],
    [0, 1, 3],
    [2, 2, 1]
])

kernel = np.array([
    [1, 0],
    [0, -1]
])

x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
k_t = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

conv = nn.Conv2d(1, 1, kernel_size=2, bias=False)
conv.weight.data = k_t

out_torch = conv(x_t)

def conv2d_multi_channel(x, kernel):
    # x: (C, H, W)
    # kernel: (C, KH, KW)
    out = 0
    for c in range(x.shape[0]):
        out += conv2d_naive(x[c], kernel[c])
    return out


print("Naive:\n", conv2d_naive(x, kernel))
print("PyTorch:\n", out_torch.squeeze().detach().numpy())


img = Image.open('../data/test.jpg').convert("L")
img = np.array(img) / 255.0

edge_kernel = np.array([
    [-1,-2,-1],
    [ 0, 0, 0],
    [ 1, 2, 1]
])

out = conv2d_naive(img, edge_kernel, padding=1)

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

plt.figure(figsize=(16,12))

# Input / Edge
plt.subplot(3,4,1)
plt.title("Input")
plt.imshow(normalize(img), cmap='gray')
plt.axis('off')

plt.subplot(3,4,2)
plt.title("Edge")
plt.imshow(normalize(conv2d_naive(img, edge_kernel, padding=1)), cmap='gray')
plt.axis('off')

# stride
for i, stride in enumerate([1,2,4,8]):
    out = conv2d_naive(img, edge_kernel, stride=stride, padding=1)

    plt.subplot(3,4,5+i)
    plt.title(f"s={stride}")
    plt.imshow(normalize(out), cmap='gray')
    plt.axis('off')

# padding
for i, padding in enumerate([0,1,5,10]):
    out = conv2d_naive(img, edge_kernel, padding=padding)

    plt.subplot(3,4,9+i)
    plt.title(f"p={padding}")
    plt.imshow(normalize(out), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.savefig("../data/all_results.png")
plt.close()
