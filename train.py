import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from model import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainDataset = torchvision.datasets.EMNIST(
    root="./", split="balanced", train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2).flip(2)),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)

model = NeuralNetwork(num_classes=47).to(device)
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for e in range(10):
    total_loss = 0
    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = lossFunction(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {e + 1}, Avg Loss: {total_loss / len(trainLoader):.4f}")

torch.save(model.state_dict(), "emnist_model.pth")