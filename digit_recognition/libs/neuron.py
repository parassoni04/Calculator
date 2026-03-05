from .utilitites import squish

class Neuron():
    def __init__(self, weights: list[float], bias: float):
        self.weights = weights
        self.bias = bias

    def computeActivation(self, activations: list[float]) -> float: 
        weightedSum: float = 0
        for i in range(0, len(self.weights)):
            weightedSum += (self.weights[i] * activations[i])

        return squish(weightedSum + self.bias)
    
    def weightsUpdation(self, weightsList: list[list[float], gradientOfCostFunction : list[float]]):
        newWeight = 0
        updatedWeights = []
        for weight,gradient in zip(weightsList,gradientOfCostFunction):
            newWeight = weight - gradient
            updatedWeights.append(newWeight)

    
