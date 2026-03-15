import numpy as np
from numpy.typing import NDArray

def squish(weightedSum: np.float64) -> np.float64:
    activation: np.float64 = 1 / (1 + np.exp(-weightedSum))
    return activation

def computeErrors(expectedOutputs: NDArray[np.uint8], computedOutputs: NDArray[np.float64]) -> NDArray[np.float64]:
    return expectedOutputs - computedOutputs

def isAccurate(expectedOutputs: NDArray[np.uint8], computedOutputs: NDArray[np.float64]) -> bool:
    for i in range(0, computedOutputs.shape[0]):
        if computedOutputs[i] < 0.5: computedOutputs[i] = 0
        else: computedOutputs[i] = 1

    if np.dot(expectedOutputs, computedOutputs) == 1: return True
    else: return False
