# external imports
import numpy as np
from numpy.typing import NDArray

class Layer:
    preSemiAvgBatchSize: int

    def __init__(self, neurons: NDArray[np.object_]):
        self.neurons: NDArray[np.object_] = neurons
        self.activations: NDArray[np.float64] = np.zeros(self.neurons.shape[0])
        self.deltas: NDArray[np.float64] = np.zeros(self.neurons.shape[0])

        self.preSemiAvgWeightGradients: NDArray[np.float64] = np.empty((self.neurons.shape[0], Layer.preSemiAvgBatchSize, self.neurons[0].weights.shape[0]), dtype=np.float64)
        self.preSemiAvgBiasGradients: NDArray[np.float64] = np.empty((self.neurons.shape[0], Layer.preSemiAvgBatchSize), dtype=np.float64)

    def computeActivtionForLayer(self, activations: NDArray[np.float64]) -> None:
        for i in range(0, self.neurons.shape[0]):
            self.activations[i] = self.neurons[i].computeActivation(activations)

    def computeDeltaForOutputLayer(self, errors: NDArray[np.float64]) -> None:
        self.deltas: NDArray[np.float64] = self.activations * (1 - self.activations) * errors

    def computeDeltaForHiddenLayer(self, deltas: NDArray[np.float64], weightMatrix: NDArray[np.float64]) -> None:
        self.deltas: NDArray[np.float64] = self.activations * (1 - self.activations) * np.dot(weightMatrix.transpose(), deltas)

    def computeGradientForLayer(self, prevActivations: NDArray[np.float64], idx: int) -> None:
        for i in range(0, self.neurons.shape[0]):
            dw, db = self.neurons[i].computeGradient(self.deltas[i], prevActivations)
            self.preSemiAvgWeightGradients[i, idx] = dw
            self.preSemiAvgBiasGradients[i, idx] = db

    def computeSemiAvgGradientsForLayer(self, idx: int) -> None:
        for i in range(0, self.neurons.shape[0]):
            self.neurons[i].setSemiAvgDwDb(self.preSemiAvgWeightGradients[i], self.preSemiAvgBiasGradients[i], idx)

    def doGradientDescentForLayer(self, learningRate: float) -> None:
        for neuron in self.neurons:
            neuron.updateWeights(learningRate)
            neuron.updateBias(learningRate)
