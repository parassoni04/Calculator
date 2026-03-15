import numpy as np
from numpy.typing import NDArray

mnistDirectory: str = "./digit_recognition/datasets/mnist"

trainImgs: NDArray[np.float64]
trainLabels: NDArray[np.uint8]
testImgs: NDArray[np.float64]
testLabels: NDArray[np.uint8]

with open(mnistDirectory + "/train-images-idx3-ubyte", "rb") as file:
    _, n, rows, columns = np.frombuffer(file.read(16), dtype=">i4")
    trainImgs = np.frombuffer(file.read(), dtype=np.uint8).reshape(n, rows, columns) / 255

with open(mnistDirectory + "/train-labels-idx1-ubyte", "rb") as file:
    _, n = np.frombuffer(file.read(8), dtype=">i4")
    trainLabels = np.frombuffer(file.read(), dtype=np.uint8).reshape(n)

with open(mnistDirectory + "/t10k-images-idx3-ubyte", "rb") as file:
    _, n, rows, columns = np.frombuffer(file.read(16), dtype=">i4")
    testImgs = np.frombuffer(file.read(), dtype=np.uint8).reshape(n, rows, columns) / 255

with open(mnistDirectory + "/t10k-labels-idx1-ubyte", "rb") as file:
    _, n = np.frombuffer(file.read(8), dtype=">i4")
    testLabels = np.frombuffer(file.read(), dtype=np.uint8).reshape(n)
