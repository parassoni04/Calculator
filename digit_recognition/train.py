# Importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transform
from model import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using MNIST dataset from torchvision to train model
trainDataset = torchvision.datasets.MNIST(
    root = "./digit_recognition",
    train = True,
    download = False,
    transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize((0.5,),(0.5,))
        ])
)

# Making batches for training
trainLoader = DataLoader(trainDataset, batch_size = 64, shuffle = True)

model = NeuralNetwork().to(device)
lossFunction = nn.CrossEntropyLoss() # Calculating Loss
optimizer = optim.Adam(model.parameters(), lr = 0.001) # Function for optimizing parameters

epochs = 10

for e in range(epochs):
    correct = 0
    total = 0
    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # Zeroes the gradient list

        output = model(images)
        predicted =output.argmax(dim=1)
        loss = lossFunction(output, labels)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loss.backward() # Calculating gradient
        optimizer.step()
    
    accuracy = (correct/total) * 100

    print(f"Epoch : {e + 1}, Loss : {loss.item() : .4f}, Accuracy : {accuracy : .2f}%")

torch.save(model.state_dict(),"./CNN/mnist_model.pth") # Saving model parameters
