# external imports
import numpy as np
from numpy.typing import NDArray

# custom imports
from libs import Neuron, Layer, computeErrors, isAccurate
from libs import testImgs, testLabels, trainImgs, trainLabels

# Hyper-parameters
rng = np.random.default_rng(37)
learningRate: float = 10
trainBatchSize: int = 1000
Layer.preSemiAvgBatchSize = 100
Neuron.semiAvgBatchSize = trainBatchSize // Layer.preSemiAvgBatchSize
testBatchSize: int = 40
numEpochs: int = 500

# Generating 128 Neurons with random weights and biases for 1st Hidden Layer
hiddenLayer_1 = Layer(np.array([Neuron(rng.standard_normal(784), np.float64(rng.standard_normal())) for i in range(0, 128)], dtype=np.object_))

# Generating 64 Neurons with random weights and biases for 2nd Hidden Layer
hiddenLayer_2 = Layer(np.array([Neuron(rng.standard_normal(128), np.float64(rng.standard_normal())) for i in range(0, 64)], dtype=np.object_))

# Generating 32 Neurons with random weights and biases for 3rd Hidden Layer
hiddenLayer_3 = Layer(np.array([Neuron(rng.standard_normal(64), np.float64(rng.standard_normal())) for i in range(0, 32)], dtype=np.object_))

# Generating 10 Neurons with random weights and biases for the Output Layer
outputLayer = Layer(np.array([Neuron(rng.standard_normal(32), np.float64(rng.standard_normal())) for i in range(0, 10)], dtype=np.object_))

# Grouping all Hidden Layers together, to make iterations easier
hiddenLayers = [
    hiddenLayer_1,
    hiddenLayer_2,
    hiddenLayer_3
]

# Training the model
for i in range(0, numEpochs):
    for j in range(0, Neuron.semiAvgBatchSize):
        for k in range(0, Layer.preSemiAvgBatchSize):
            # index of the training set
            trainIdx: int = (i * trainBatchSize + j * Layer.preSemiAvgBatchSize + k) % trainImgs.shape[0]

            # Feed Forward: Computing Activations for the 1st Hidden Layer
            hiddenLayers[0].computeActivtionForLayer(trainImgs[trainIdx].flatten())

            # Feed Forward: Computing Activations for the Hidden Layers other than the 1st one
            for k in range(1, len(hiddenLayers)):
                hiddenLayers[k].computeActivtionForLayer(hiddenLayers[k-1].activations)

            # Feed Forward: Computing Activations for the Output Layer
            outputLayer.computeActivtionForLayer(hiddenLayers[len(hiddenLayers) - 1].activations)

            # Feed Forward: Computing individual costs for all neurons in Output Layer
            expectedOutputs: NDArray[np.uint8] = np.zeros(10, dtype=np.uint8)
            expectedOutputs[trainLabels[trainIdx]] = 1

            errors: NDArray[np.float64] = computeErrors(expectedOutputs, outputLayer.activations)

            # Back Propagation: Computing Deltas for Output Layer
            outputLayer.computeDeltaForOutputLayer(errors)

            # Back Propagation: Computing Deltas for last Hidden Layer
            outputLayerWeightMatrix: NDArray[np.float64] = np.stack([neuron.weights for neuron in outputLayer.neurons])
            hiddenLayers[len(hiddenLayers) - 1].computeDeltaForHiddenLayer(outputLayer.deltas, outputLayerWeightMatrix)

            # Back Propagation: Computing Deltas for remaining Hidden Layers
            for k in range(len(hiddenLayers) - 2, -1, -1):
                weightsMatrix: NDArray[np.float64] = np.stack([neuron.weights for neuron in hiddenLayers[k+1].neurons])
                hiddenLayers[k].computeDeltaForHiddenLayer(hiddenLayers[k+1].deltas, weightsMatrix)

            # Back Propagation: Computing Gradients for weights and biases of all Layers
            hiddenLayers[0].computeGradientForLayer(trainImgs[trainIdx].flatten(), k)
            for k in range(1, len(hiddenLayers)):
                hiddenLayers[k].computeGradientForLayer(hiddenLayers[k-1].activations, k)
            outputLayer.computeGradientForLayer(hiddenLayers[len(hiddenLayers) - 1].activations, k)
        
        # Averaging the stored weight/bias gradients prematurely to lower memory usage
        for layer in hiddenLayers:
            layer.computeSemiAvgGradientsForLayer(j)

    # Back Propagation: Adjusting all the weights and biases according to the average of gradients
    for layer in hiddenLayers:
        layer.doGradientDescentForLayer(learningRate)
    outputLayer.doGradientDescentForLayer(learningRate)

    # Testing: Computing accuracy in terms of SumOfSquaredErrors and CorrectPredicions
    sse: np.float64 = np.float64(0)
    accurateOutputs: int = 0

    for j in range(0, testBatchSize):
        # Feed Forward: Computing Activations for the 1st Hidden Layer
        hiddenLayers[0].computeActivtionForLayer(testImgs[j].flatten())

        # Feed Forward: Computing Activations for the Hidden Layers other than the 1st one
        for k in range(1, len(hiddenLayers)):
            hiddenLayers[k].computeActivtionForLayer(hiddenLayers[k-1].activations)

        # Feed Forward: Computing Activations for the Output Layer
        outputLayer.computeActivtionForLayer(hiddenLayers[len(hiddenLayers) - 1].activations)

        # Feed Forward: Computing individual costs for all neurons in Output Layer
        expectedOutputs: NDArray[np.uint8] = np.zeros(10, dtype=np.uint8)
        expectedOutputs[testLabels[j]] = 1

        errors: NDArray[np.float64] = computeErrors(expectedOutputs, outputLayer.activations)
        sse += np.mean(errors ** 2)
        
        accurate: bool = isAccurate(expectedOutputs, outputLayer.activations)
        accurateOutputs += accurate

    accuracy: float = accurateOutputs / testBatchSize
    sse /= testBatchSize
    print(f"[{i}th]: Cost: {sse}, Accuracy: {accuracy}")
