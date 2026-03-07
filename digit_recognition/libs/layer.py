try :
    from neuron import Neuron
    from utilities import transpose
except :
    from .neuron import Neuron
    from .utilities import transpose

class Layer:
    def __init__(self, neuronList: list[Neuron]):
        self.neurons: list[Neuron] = neuronList
        self.activations : list[float]  = []
        self.deltaList: list[float]= []

    def computeActivtionForLayer(self, activations: list[float]) -> None:
        self.activations= []
        for neuron in self.neurons:
            activation: float = neuron.computeActivation(activations)
            self.activations.append(activation)

    def computeDeltaForOutputLayer(self, costs: list[float]):
        self.deltaList: list[float] = []
        for cost, activation in zip(costs, self.activations):
            tempResult: float = activation*(1-activation)*(2*cost)
            self.deltaList.append(tempResult)

    def computeDeltaForHiddenLayer(self, deltas: list[float], weightMatrix: list[list[float]]):
        self.deltaList = []
        wDSum : float = 0
        transposedWeightMatrix = transpose(weightMatrix)

        for weights in transposedWeightMatrix:
            wDSum : float = 0
            for weight, delta in zip(weights, deltas):
                wD: float = weight*delta
                wDSum: float = wDSum + wD
            for activation in self.activations:
                tempResult: float = activation*(1-activation)*wDSum
                self.deltaList.append(tempResult)

    def computeGradientForLayer(self, prevActivations: list[float]):
        for delta, neuron in zip(self.deltaList, self.neurons):
            neuron.computeGradient(delta, prevActivations)

    def doGradientDecentForLayer(self, learningRate: float):
        for neuron in self.neurons:
            neuron.updateWeights(learningRate)
            neuron.updateBias(learningRate)
