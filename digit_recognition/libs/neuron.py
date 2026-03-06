try : 
    from utilitites import squish, transpose
except:
    from .utilitites import squish, transpose

class Neuron():
    def __init__(self, weights: list[float], bias: float):
        self.weights = weights
        self.bias = bias
        self.allWeightsGradients : list[list[float]] = []
        self.allBiasesGradients : list[float] = []

    def computeActivation(self, activations: list[float]) -> float: 
        weightedSum: float = 0
        for i in range(0, len(self.weights)):
            weightedSum += (self.weights[i] * activations[i])

        return squish(weightedSum + self.bias)
    
    def computeGradient(self, delta : float, prevActivations : list[float]) -> None:
        weightGradient : list[float] = []
        for activation in prevActivations:
            gradient = delta * activation
            weightGradient.append(gradient)

        biasGradient : float = delta

        self.allWeightsGradients.append(weightGradient)
        self.allBiasesGradients.append(biasGradient)

    def updateWeights(self) -> None:
        self.allWeightsGradients = transpose(self.allWeightsGradients)
        weightGradient : list[float] = []
        averageWeights : float = 0
        for gradients in self.allWeightsGradients:
            averageWeights = 0
            for gradient in gradients:
                averageWeights += gradient
            averageWeights = averageWeights/len(gradients)
            weightGradient.append(averageWeights)

        for gradient in weightGradient:
            self.weights = self.weights - gradient

        self.allWeightsGradients = []

    def updateBias(self) -> None:
        for biasGradient in self.allBiasesGradients:
            self.bias -= biasGradient

        self.allBiasesGradients = []

        