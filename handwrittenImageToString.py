import torch #type: ignore
import numpy as np
from segmentation import segment
from model import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convertHandwrittenToString(imagePath: str = "./sampleimg.jpg", modelPath: str = "mnist_model.pth") -> str:
    segmentedCrops : list[np.ndarray] = segment(imagePath)
    
    
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(modelPath,map_location = device, weights_only = False))
    model.eval()
    
    predictions : list = []
    
    with torch.no_grad():
        for crop in segmentedCrops:
            segmentedTensor  =  torch.tensor(crop,dtype=torch.float32)/255.0
            segmentedTensor = segmentedTensor.unsqueeze(0).unsqueeze(0)
            segmentedTensor = segmentedTensor.to(device)
            output = model(segmentedTensor)
            pred = str(output.argmax(dim=1).item())
            print(pred)
    expresion = "".join(predictions)
    print(expresion)
    return expresion

if __name__ == "__main__":
    convertHandwrittenToString()