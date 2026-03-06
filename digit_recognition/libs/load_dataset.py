from mnist import MNIST

mnistDirectory: str = "./digit_recognition/datasets/mnist"
data = MNIST(mnistDirectory)
trainImgs, trainLabels = data.load_training()
testImgs, testLabels = data.load_testing()

for img in trainImgs:
    for i in range(0, len(img)):
        img[i] = img[i] / 255

for img in testImgs:
    for i in range(0, len(img)):
        img[i] = img[i] / 255
