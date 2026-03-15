# External imports
import numpy as np
from numpy.typing import NDArray

# Custom imports
try:
    from utilities import squish
except:
    from .utilities import squish

class Neuron():
    semiAvgBatchSize: int

    def __init__(self, weights: NDArray[np.float64], bias: np.float64):
        self.weights = weights
        self.bias = bias

        self.semiAvgWeightGradients: NDArray[np.float64] = np.empty((Neuron.semiAvgBatchSize, self.weights.shape[0]), dtype=np.float64)
        self.semiAvgBiasGradients: NDArray[np.float64] = np.empty(Neuron.semiAvgBatchSize, dtype=np.float64)

    def computeActivation(self, activations: NDArray[np.float64]) -> np.float64:
        return squish(np.dot(self.weights, activations) + self.bias)

    def computeGradient(self, delta: np.float64, prevActivations: NDArray[np.float64]) -> tuple[NDArray[np.float64], np.float64]:
        dw: NDArray[np.float64] = np.dot(delta, prevActivations) # basically a scalar multiplication of a 1darray
        db: np.float64 = delta # aliasing for the sake of readability

        return (dw, db)

    def setSemiAvgDwDb(self, dW: NDArray[np.float64], dB: NDArray[np.float64], idx: int) -> None:
        self.semiAvgWeightGradients[idx] = dW.mean(axis=0) # averages values along the columns in a 2darray and returns a 1darray
        self.semiAvgBiasGradients[idx] = dB.mean()

    def updateWeights(self, learningRate: float) -> None:
        fullyAvgWeightGradients: NDArray[np.float64] = self.semiAvgWeightGradients.mean(axis=0)
        self.weights += np.dot(learningRate, fullyAvgWeightGradients)

    def updateBias(self, learningRate: float) -> None:
        fullyAvgBiasGradient: np.float64 = self.semiAvgBiasGradients.mean()
        self.bias += np.multiply(learningRate, fullyAvgBiasGradient)
