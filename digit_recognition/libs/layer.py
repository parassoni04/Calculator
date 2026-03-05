
class Layer:
    def __init__(self, neuronList: list[Neuron]):
        self.neurons: list[Neuron] = neuronList
        self.activations : list[float]  = []
    
    def computeActivtionForLayer(self, activations: list[float]) -> None:
        self.activations= []
        for neuron in self.neurons:
            activation: float = neuron.computeActivation(activations)
            self.activations.append(activation)