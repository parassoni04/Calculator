
class Layer:
    def __init__(self, neuronList: list[Neuron]):
        self.layer: list[Neuron] = neuronList
    
    def computeActivtionForLayer(self, activation: float) -> None:
        temp: list[float] = []
        for neuron in self.layer:
            temp.append(neuron.computeActivation(activation))
        