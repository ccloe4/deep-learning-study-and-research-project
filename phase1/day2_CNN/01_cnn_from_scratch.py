import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

############################################
# 1. NAIVE CONV2D (FORWARD ONLY)
############################################

def conv2d_naive(x, w, b, stride=1, padding=0):
    N, C, H, W = x.shape
    F, _, KH, KW = w.shape

    H_out = (H - KH + 2 * padding) // stride + 1
    W_out = (W - KW + 2 * padding) // stride + 1

    x_padded = np.pad(
        x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant'
    )

    out = np.zeros((N, F, H_out, W_out))

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride

                    region = x_padded[n, :, h_start:h_start+KH, w_start:w_start+KW]
                    out[n, f, i, j] = np.sum(region * w[f]) + b[f]

    return out


############################################
# 2. NUMPY LAYERS
############################################

def relu(x):
    return np.maximum(0, x)


def maxpool2d(x, pool_size=2, stride=2):
    N, C, H, W = x.shape

    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride

                    region = x[n, c, h_start:h_start+pool_size, w_start:w_start+pool_size]
                    out[n, c, i, j] = np.max(region)

    return out


############################################
# 3. NUMPY CNN (FORWARD ONLY)
############################################

def numpy_cnn_forward(x):
    # random weights (검증용)
    w1 = np.random.randn(16, 3, 3, 3)
    b1 = np.zeros(16)

    w2 = np.random.randn(32, 16, 3, 3)
    b2 = np.zeros(32)

    # Conv → ReLU → Pool
    x = conv2d_naive(x, w1, b1, padding=1)
    x = relu(x)
    x = maxpool2d(x)

    # Conv → ReLU → Pool
    x = conv2d_naive(x, w2, b2, padding=1)
    x = relu(x)
    x = maxpool2d(x)

    return x


############################################
# 4. PYTORCH MODEL (ACTUAL TRAINING)
############################################

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 → 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 → 8x8

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


############################################
# 5. DATASET (CIFAR-10)
############################################

def get_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader


############################################
# 6. TRAIN LOOP
############################################

def train(model, trainloader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(10):
        total_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


############################################
# 7. EVALUATION
############################################

def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")


############################################
# 8. MAIN
############################################

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # numpy sanity check
    x = np.random.randn(1, 3, 32, 32)
    out = numpy_cnn_forward(x)
    print("NumPy CNN output shape:", out.shape)  # (1, 32, 8, 8)

    # training
    trainloader, testloader = get_dataloader()

    model = SimpleCNN().to(device)

    train(model, trainloader, device)
    evaluate(model, testloader, device)