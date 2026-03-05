
class Layer:
    def __init__(self, neuronList: list[Neuron]):
        self.neurons: list[Neuron] = neuronList
        self.activations : list[float]  = []
    
    def computeActivtionForLayer(self, activation: list[float]) -> None:
        for neuron in self.neurons:
            tempActivation: float = neuron.computeActivation(activation)
            self.activations.append(tempActivation)
        