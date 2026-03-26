# Importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transform
from model import NeuralNetwork
from torch.utils.data import random_split

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

trainSize = 50000
valSize = 10000

trainDataset, valDataset = random_split(trainDataset, [trainSize, valSize])

# Making batches for training and validation
trainLoader = DataLoader(trainDataset, batch_size = 64, shuffle = True)
valLoader = DataLoader(valDataset, batch_size = 64, shuffle = False)

model = NeuralNetwork().to(device)
lossFunction = nn.CrossEntropyLoss() # Calculating Loss
optimizer = optim.Adam(model.parameters(), lr = 0.001) # Function for optimizing parameters

epochs = 10

for e in range(epochs):

    # Training
    model.train()
    trainCorrect = 0
    trainTotal = 0

    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # Zeroes the gradient list
        output = model(images)
        loss = lossFunction(output, labels)

        loss.backward() # Calculating gradient
        optimizer.step()

        predicted =output.argmax(dim=1)
        trainCorrect += (predicted == labels).sum().item()
        trainTotal += labels.size(0)

    trainAcc = (trainCorrect/trainTotal) * 100
    
    # Validation
    model.eval()
    valCorrect = 0
    valTotal = 0

    with torch.no_grad():
        for images, labels in valLoader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            predicted = output.argmax(dim=1)

            valCorrect += (predicted == labels).sum().item()
            valTotal += labels.size(0)

    valAcc = (valCorrect/valTotal) * 100



    print(f"Epoch : {e + 1}, Loss : {loss.item() : .4f}, Train Accuracy : {trainAcc : .2f}%, Validation Accuracy : {valAcc : .2f}%")

torch.save(model.state_dict(),"./digit_recognition/mnist_model.pth") # Saving model parameters
