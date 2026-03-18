import torch#type:ignore
import torch.nn as nn#type:ignore
import torch.nn.functional as F#type:ignore
import torch.optim as optium#type:ignore
from torch.utils.data import DataLoader#type:ignore
import torchvision#type:ignore
import torchvision.transforms as transform#type:ignore
from model import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainDataset = torchvision.datasets.MNIST(
    root = "./",
    train = True,
    download = False    ,
    transform = transform.ToTensor()
)

trainLoader = DataLoader(trainDataset, batch_size = 100, shuffle = True)

model = NeuralNetwork().to(device)
lossFunction = nn.CrossEntropyLoss()
optimizer = optium.Adam(model.parameters(), lr = 0.001)

epochs = 10

for e in range(epochs):
    for images, labels in trainLoader:
        images , labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        output = model(images)
        loss = lossFunction(output, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch : {e + 1}, Loss : {loss.item()}")

torch.save(model.state_dict(),"mnist_model.pth")
