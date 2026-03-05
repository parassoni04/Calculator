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
        self.gradientList = list[list[float]] = [[]]
        self.gradientBais: list[float] = 0
        
        
    
    def computeActivtionForLayer(self, activations: list[float]) -> None:
        self.activations= []
        for neuron in self.neurons:
            activation: float = neuron.computeActivation(activations)
            self.activations.append(activation)
    
    def computeDeltaForOutputLayer(self, costs: list[float]):
        self.deltaList: list[float] = []
        for cost, activation in zip(costs, self.activations):
            tempResult:float = activation*(1-activation)*(2*cost)
            self.deltaList.append(tempResult)
            
    def computeDeltaForHiddenLayer(self, deltas: list[float], weightsList: list[list[float]]):
        self.deltaList = []
        wDSum : float = 0
        transposeLists = [list(row) for row in zip(*weightsList)]

        for  weights, delta in zip(transposeLists, deltas):
            wDSum : float = 0
            for weight in weights:
                wD: float = weight*delta
                wDSum: float = wDSum + wD
            for activation in self.activations:
                tempResult: float = activation*(1-activation)*wDSum
                self.deltaList.append(tempResult)
            
    def computeGradientForLayer(self, prevActivations: list[float], learningRate: float):
        self.gradientList = [[]]
        self.gradientBais = []
        tempList: list[float] = []
        for delta in self.deltaList:
            for activation  in prevActivations:
                wD: float = learningRate * activation * delta
                tempList.append(wD)
                self.gradientList.append(tempList)
        for delta in self.deltaList:
            bais: float = learningRate*delta
            self.gradientBais.append(bais)
            
