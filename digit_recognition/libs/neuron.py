from utilitites import squish

class Neuron():
    def __init__(self, weights : list[float], bias : float):
        self.weights = weights
        self.bias = bias

    def computeActivation(self, weights : list[float], bias : float, activation : float):
        for weight in weights:
            weightedSum : float = (weight * self.activation)

        squish(weightedSum + bias)