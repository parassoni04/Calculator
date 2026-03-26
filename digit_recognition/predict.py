# Importing libraries
from model import NeuralNetwork
import torch
import torchvision
import torchvision.transforms as transform
from torch.utils.data import DataLoader

# Converting to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using MNIST dataset from torchvision to test model 
testDataset = torchvision.datasets.MNIST(
    root = "./digit_recognition",
    train = False,
    download = False,
    transform = transform.ToTensor()
)

# Making batches for testing
testLoader = DataLoader(testDataset, batch_size = 1000, shuffle = False)

path = "./digit_recognition/mnist_model.pth"

model = NeuralNetwork().to(device)

# Loading saved parameters
model.load_state_dict(torch.load(path,map_location = device,weights_only = False))
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images , labels in testLoader:
        images , labels = images.to(device),labels.to(device)
        
        output = model(images)
        predicted =output.argmax(dim=1)
        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    accuracy : float = (correct/total) * 100

print(f"Accuracy : {accuracy : .2f}%")