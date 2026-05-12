from model import NeuralNetwork

def getParamCount(model: NeuralNetwork) -> int:
    paramCount: int = 0
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        paramCount += param.numel()

    return paramCount
