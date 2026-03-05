from libs import Neuron, Layer, computeCosts
import random

# Random input for testing purposes only
inputActivations: list[float] = [random.random() for i in range(0, 728)]

# Generating 128 Neurons with random weights and biases for 1st Hidden Layer
hiddenLayer_1 = Layer([
    Neuron([random.uniform(-1, 1) for i in range(0, 728)], random.uniform(-1, 1))
    for i in range(0, 128)
])

# Generating 64 Neurons with random weights and biases for 2nd Hidden Layer
hiddenLayer_2 = Layer([
    Neuron([random.uniform(-1, 1) for i in range(0, 128)], random.uniform(-1, 1))
    for i in range(0, 64)
])

# Generating 32 Neurons with random weights and biases for 3rd Hidden Layer
hiddenLayer_3 = Layer([
    Neuron([random.uniform(-1, 1) for i in range(0, 64)], random.uniform(-1, 1))
    for i in range(0, 32)
])

# Generating 10 Neurons with random weights and biases for the Output Layer
outputLayer = Layer([
    Neuron([random.uniform(-1, 1) for i in range(0, 32)], random.uniform(-1, 1))
    for i in range(0, 10)
])

# Grouping all Hidden Layers together, to make iterations easier
hiddenLayers: list[Layer] = [
    hiddenLayer_1,
    hiddenLayer_2,
    hiddenLayer_3
]

# Feed Forward: Computing Activations for the 1st Hidden Layer
hiddenLayers[0].computeActivtionForLayer(inputActivations)


# Feed Forward: Computing Activations for the Hidden Layers other than the 1st one
for i in range(1, len(hiddenLayers)):
    hiddenLayers[i].computeActivtionForLayer(hiddenLayers[i-1].activations)

# Feed Forward: Computing Activations for the Output Layer
lastHiddenLayerActivations: list[float] = hiddenLayers[len(hiddenLayers) - 1].activations
outputLayer.computeActivtionForLayer(lastHiddenLayerActivations)
print("Output: ", outputLayer.activations)

# Feed Forward: Computing individual costs for all neurons in Output Layer
# Random expectedOutputs, for testing purposes only
expectedOutputs: list[float] = [0 for i in range(0, 10)]
costs: list[float] = computeCosts(expectedOutputs, outputLayer.activations)
print("Costs: ", costs)
