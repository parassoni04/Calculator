import torch
import numpy as np
import torchvision.transforms as transforms
from segmentation import segment
from model import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMNIST_BALANCED_MAP = (
    [str(i) for i in range(10)] +
    [chr(i) for i in range(ord('A'), ord('Z') + 1)] +
    list('abdefghnqrt')
)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def convertHandwrittenToString(
    imagePath: str = "./Images/sampleimg.webp",
    modelPath: str = "emnist_model.pth"
) -> str:
    segmentedCrops: list[np.ndarray] = segment(imagePath)

    model = NeuralNetwork(num_classes=47).to(device)
    model.load_state_dict(torch.load(modelPath, map_location=device, weights_only=False))
    model.eval()

    predictions: list = []

    with torch.no_grad():
        for crop in segmentedCrops:
            tensor = preprocess(crop).unsqueeze(0).to(device)
            output = model(tensor)
            idx = output.argmax(dim=1).item()
            predictions.append(EMNIST_BALANCED_MAP[idx])

    expression = "".join(predictions)
    print(expression)
    return expression

if __name__ == "__main__":
    convertHandwrittenToString()