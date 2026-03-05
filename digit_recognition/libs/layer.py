
class Layer:
    def __init__(self, neuronList: list[Neuron]):
        self.neurons: list[Neuron] = neuronList
        self.activations : list[float]  = []
    
    def computeActivtionForLayer(self, activations: list[float]) -> None:
        for neuron in self.neurons:
            self.activations= []
            activation: float = neuron.computeActivation(activations)
            self.activations.append(activation)