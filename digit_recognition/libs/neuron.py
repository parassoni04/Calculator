try : 
    from utilities import squish
except:
    from .utilities import squish

class Neuron():
    def __init__(self, weights: list[float], bias: float):
        self.weights = weights
        self.bias = bias

    def computeActivation(self, activations: list[float]) -> float: 
        weightedSum: float = 0
        for i in range(0, len(self.weights)):
            weightedSum += (self.weights[i] * activations[i])

        return squish(weightedSum + self.bias)
    
    def weightsUpdation(self, weightGradients : list[float]) -> None:
        newWeight : float = 0
        updatedWeights : list[float] = []
        for weight, gradient in zip(self.weights, weightGradients):
            newWeight = weight - gradient
            updatedWeights.append(newWeight)

        self.weights = updatedWeights

    
    def biasUpdation(self, biasGradient : float) -> None:
        self.bias -= biasGradient
        