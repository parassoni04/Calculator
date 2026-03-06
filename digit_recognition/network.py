from libs import Neuron, Layer, computeCosts
from libs import testImgs, testLabels, trainImgs, trainLabels
import random

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

# Hyper-parameters
learningRate: float = 0.001
numberOfTrainingElements: int = 400
numberOfTestingElements: int = 40

# Training the model
for i in range(0, 10):
    for j in range(0, numberOfTrainingElements):
        # Feed Forward: Computing Activations for the 1st Hidden Layer
        hiddenLayers[0].computeActivtionForLayer(trainImgs[j])

        # Feed Forward: Computing Activations for the Hidden Layers other than the 1st one
        for k in range(1, len(hiddenLayers)):
            hiddenLayers[k].computeActivtionForLayer(hiddenLayers[k-1].activations)

        # Feed Forward: Computing Activations for the Output Layer
        lastHiddenLayerActivations: list[float] = hiddenLayers[len(hiddenLayers) - 1].activations
        outputLayer.computeActivtionForLayer(lastHiddenLayerActivations)
        # print("Output: ", outputLayer.activations)

        # Feed Forward: Computing individual costs for all neurons in Output Layer
        expectedOutputs: list[float] = [0 for k in range(0, 10)]
        expectedOutputs[trainLabels[j]] = 1

        costs: list[float] = computeCosts(expectedOutputs, outputLayer.activations)
        # print("Costs: ", costs)

        # Back Propagation: Computing Deltas for Output Layer
        outputLayer.computeDeltaForOutputLayer(costs)

        # Back Propagation: Computing Deltas for last Hidden Layer
        outputLayerWeightMatrix: list[list[float]] = [neuron.weights for neuron in outputLayer.neurons]
        hiddenLayers[len(hiddenLayers) - 1].computeDeltaForHiddenLayer(outputLayer.deltaList, outputLayerWeightMatrix)

        # Back Propagation: Computing Deltas for every other Hidden Layer
        for k in range(0, len(hiddenLayers) - 2):
            weightsMatrix: list[list[float]] = [neuron.weights for neuron in hiddenLayers[k+1].neurons]
            hiddenLayers[k].computeDeltaForHiddenLayer(hiddenLayers[k+1].deltaList, weightsMatrix)

        # Back Propagation: Computing Gradients for weights and biases of all Layers
        hiddenLayers[0].computeGradientForLayer(trainImgs[0])
        for k in range(1, len(hiddenLayers)):
            hiddenLayers[k].computeGradientForLayer(hiddenLayers[k-1].activations)
        outputLayer.computeGradientForLayer(hiddenLayers[len(hiddenLayers) - 1].activations)

    # Back Propagation: Adjusting all the weights and biases according to the average of gradients
    for layer in hiddenLayers:
        layer.doGradientDecentForLayer(learningRate)
    outputLayer.doGradientDecentForLayer(learningRate)

    # Testing: Computing accuracy in terms of SumOfSquaredErrors
    for j in range(0, numberOfTestingElements):
        # Feed Forward: Computing Activations for the 1st Hidden Layer
        hiddenLayers[0].computeActivtionForLayer(testImgs[j])

        # Feed Forward: Computing Activations for the Hidden Layers other than the 1st one
        for k in range(1, len(hiddenLayers)):
            hiddenLayers[k].computeActivtionForLayer(hiddenLayers[k-1].activations)

        # Feed Forward: Computing Activations for the Output Layer
        lastHiddenLayerActivations: list[float] = hiddenLayers[len(hiddenLayers) - 1].activations
        outputLayer.computeActivtionForLayer(lastHiddenLayerActivations)
        print("Output: ", outputLayer.activations)

        # Feed Forward: Computing individual costs for all neurons in Output Layer
        expectedOutputs: list[float] = [0 for k in range(0, 10)]
        expectedOutputs[testLabels[j]] = 1

        costs: list[float] = computeCosts(expectedOutputs, outputLayer.activations)
        sse: float = 0
        for k in range(0, len(costs)):
            sse += costs[k] ** 2
        
        print(f"{i}th Cost: {sse}")
