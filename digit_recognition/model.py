# Importing Libraries
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Initializing Neural Network

        self.dropout = nn.Dropout(0.3)

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool  = nn.MaxPool2d(2, 2)
        
        self.fc1   = nn.Linear(64 * 5 * 5, 256)
        self.fc2   = nn.Linear(256, 128)
        self.fc3   = nn.Linear(128, 64)
        self.fc4   = nn.Linear(64, 62)

    def forward(self, x):

        # Calculating feed forward

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x