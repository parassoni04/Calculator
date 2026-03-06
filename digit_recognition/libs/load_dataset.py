from mnist import MNIST

mnistDirectory: str = "./digit_recognition/datasets/mnist"
data = MNIST(mnistDirectory)
train_imgs, train_labels = data.load_training()
test_imgs, test_labels = data.load_testing()

for img in train_imgs:
    for i in range(0, len(img)):
        img[i] = img[i] / 255

for img in test_imgs:
    for i in range(0, len(img)):
        img[i] = img[i] / 255
