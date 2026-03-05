from .neuron import Neuron
class Layer:
    def __init__(self, neuronList: list[Neuron]):
        self.neurons: list[Neuron] = neuronList
        self.activations : list[float]  = []
        self.hiddenDeltaList: list[float]= []
        self.outputDeltaList: list[float]= []
        
    
    def computeActivtionForLayer(self, activations: list[float]) -> None:
        self.activations= []
        for neuron in self.neurons:
            activation: float = neuron.computeActivation(activations)
            self.activations.append(activation)
    
    def computeDeltaForOutputLayer(self, costs: list[float]):
        self.outputDeltaList: list[float] = []
        for cost , activation in zip(cost , self.activations):
            tempResult:float = activation*(1-activation)*(2*cost)
            self.outputDeltaList.append(tempResult)
            
    def computeDeltaForHiddenLayer(self, deltas: list[float], weightsList: list[list[float]]):
        self.hiddenDeltaList = []
        wDSum : float = 0
        for  weights in weightsList:
            for weight, delta in zip(weights, deltas):
                wD: float = weight*delta
                wDSum: float = wDSum + wD
    
        for activation in self.activations:
            tempResult: float = activation*(1-activation)*wDSum
            self.hiddenDeltaList.append(tempResult)