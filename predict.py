from model import NeuralNetwork
import torch#type:ignore
import torchvision#type:ignore
import torchvision.transforms as transform#type:ignore
from torch.utils.data import DataLoader#type:ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testDataset = torchvision.datasets.MNIST(
    root = "./",
    train = False,
    download = False,
    transform = transform.ToTensor()
)
testLoader = DataLoader(testDataset, batch_size = 1000, shuffle = False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "mnist_model.pth"

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(path,map_location = device,weights_only = True))
model.eval()

with torch.no_grad():
    for images , lables in testLoader:
        images , lables = images.to(device),lables.to(device)
        
        output = model(images)
        predicted =output.argmax(dim=1)
        
        for i in range(1000):
            print(f"Predicted : {predicted[i].item() }  | True : {lables[i].item()}")
        break
