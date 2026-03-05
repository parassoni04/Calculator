from utilitites import squish

class Neuron():
    def __init__(self, weights: list[float], bias: float):
        self.weights = weights
        self.bias = bias

    def computeActivation(self, activation: list) -> float: 
        weightedSum: float = 0
        for i in range(0, len(activation)):
            weightedSum += (self.weights[i] * activation[i])

        return squish(weightedSum + self.bias)