import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optium
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transform

trainDataset = torchvision.datasets.MNIST(
    root = "./digit_recognition",
    train = True,
    download = False,
    transform = transform.ToTensor()
)

testDataset = torchvision.datasets.MNIST(
    root = "./digit_recognition",
    train = False,
    download = False,
    transform = transform.ToTensor()
)

trainLoader = DataLoader(trainDataset, batch_size = 60000, shuffle = True)
testLoader = DataLoader(testDataset, batch_size = 100, shuffle = False)

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32,64,3)

        self.pool = nn.MaxPool2d(2,2) # Downscale

        self.fc1 = nn.Linear(64*5*5,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1) # Done for flattening ([batch, 64, 5, 5] -> [batch, 1600])

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = (self.fc4(x))

        return x
    
model = NeuralNetwork()

lossFunction = nn.CrossEntropyLoss()
optimizer = optium.Adam(model.parameters(), lr = 0.001)

epochs = 10

for e in range(epochs):
    for images, labels in trainLoader:
        optimizer.zero_grad()

        output = model(images)
        loss = lossFunction(output, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch : {e + 1}, Loss : {loss.item()}")

