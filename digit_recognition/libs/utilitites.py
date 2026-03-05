def squish(weightedSum: float) -> float:
    e: float = 2.718
    exponentiatedWeightedSum: float = e ** (-1 * weightedSum)
    activation: float = 1 / (1 + exponentiatedWeightedSum)

    return activation