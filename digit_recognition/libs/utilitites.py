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
        squaredError = error ** 2
        costs.append(squaredError)
    
    return costs

def transpose(matrix: list[list[float]]) -> list[list[float]]:
    transpose: list[list[float]] = []

    for i in range(0, len(matrix[0])):
        transposedRow: list[float] = []
        for j in range(0, len(matrix)):
            transposedRow.append(matrix[j][i])
        transpose.append(transposedRow)

    return transpose
