from utilitites import squish

class Neuron():
    def __init__(self, weights : list[float], bias : float):
        self.weights = weights
        self.bias = bias

    def computeActivation(self, activation : float):
        for weight in self.weights:
            weightedSum : float = (weight * self.activation)

        squish(weightedSum + self.bias)