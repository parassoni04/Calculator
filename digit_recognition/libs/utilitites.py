import math

def squish(weightedSum: float) -> float:
    e: float = math.e
    exponentiatedWeightedSum: float = e ** (-1 * weightedSum)
    activation: float = 1 / (1 + exponentiatedWeightedSum)

    return activation

def computeCosts(expectedOutputs: list[float], computedOutputs: list[float]) -> list[float]:
    costs: list[float] = []
    for i in range(0, len(expectedOutputs)):
        error: float = expectedOutputs[i] - computedOutputs[i]
        costs.append(error)
    
    return costs
