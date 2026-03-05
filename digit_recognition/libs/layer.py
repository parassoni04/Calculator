
class Layer:
    def __init__(self,neuronList):
        self.layer:list[Neuron]=neuronList
    
    def computeActivtionForLayer(self)-> list[Neuron]:
        temp:list[float]
        for n in self.layer:
            temp.append(n.computeActivation())
        return temp