
class Layer:
    def __init__(self,neuronList):
        self.layer: list[Neuron] = neuronList
    
    def computeActivtionForLayer(self) -> None:
        temp: list[float] = []
        for n in self.layer:
            temp.append(n.computeActivation())
        