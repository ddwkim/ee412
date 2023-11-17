import csv
import sys
import math
import numpy as np

np.random.seed(0)

LR = 1
NUM_EPOCHS = 4
BSZ = 64
DEBUG = False


class Fully_Connected_Layer:
    def __init__(self, learning_rate):
        self.InputDim = 784
        self.HiddenDim = 256
        self.OutputDim = 10
        self.learning_rate = learning_rate

        """Weight Initialization"""
        self.W1 = np.random.uniform(
            -1 / math.sqrt(self.InputDim),
            1 / math.sqrt(self.InputDim),
            (self.InputDim, self.HiddenDim),
        )
        self.b1 = np.zeros((self.HiddenDim))
        self.W2 = np.random.uniform(
            -1 / math.sqrt(self.HiddenDim),
            1 / math.sqrt(self.HiddenDim),
            (self.HiddenDim, self.OutputDim),
        )
        self.b2 = np.zeros((self.OutputDim))

    def Forward(self, Input):
        """Implement forward propagation"""
        z1 = Input @ self.W1 + self.b1
        h = np.maximum(z1, 0)
        z2 = h @ self.W2 + self.b2
        o = sigmoid(z2)

        return o, z2, h, z1

    def Backward(self, Input, Label, Output):
        """Implement backward propagation"""
        """Update parameters using gradient descent"""
        o, z2, h, z1 = Output

        dL_dW2 = np.matmul(h.T, (o - Label) * o * (1 - o))
        dL_db2 = np.sum((o - Label) * o * (1 - o), axis=0)

        dL_dW1 = np.matmul(
            Input.T,
            np.matmul((o - Label) * o * (1 - o), self.W2.T)
            * (z1 > 0),
        )
        dL_db1 = np.sum(
            np.matmul((o - Label) * o * (1 - o), self.W2.T)
            * (z1 > 0),
            axis=0,
        )

        self.W2 -= self.learning_rate * dL_dW2 / len(Label)
        self.b2 -= self.learning_rate * dL_db2 / len(Label)
        self.W1 -= self.learning_rate * dL_dW1 / len(Label)
        self.b1 -= self.learning_rate * dL_db1 / len(Label)

    def Train(self, Input, Label):
        Output = self.Forward(Input)
        if DEBUG:
            accuracy = np.sum(
                np.argmax(Output[0], axis=1)
                == np.argmax(Label, axis=1)
            ) / len(Label)
            print(f"Train Accuracy: {accuracy:.4f}")

        self.Backward(Input, Label, Output)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def one_hot_encoding(label, num_classes):
    encoded_label = np.zeros((len(label), num_classes))
    for i in range(len(label)):
        encoded_label[i][label[i]] = 1
    return encoded_label


def read_mnist_data(filename):
    data = []
    label = []
    with open(filename, "rt") as f:
        reader = csv.reader(f)
        for row in reader:
            label.append(int(float(row[-1])))
            data.append(list(map(float, row[:-1])))
    data = np.array(data)
    label = np.array(
        one_hot_encoding(label, max(label) + 1)
    )

    return data, label


def main():
    """Load MNIST data"""

    train_data, train_label = read_mnist_data(sys.argv[1])
    test_data, test_label = read_mnist_data(sys.argv[2])

    """Construct a fully-connected network"""
    Network = Fully_Connected_Layer(LR)

    """Train the network for the number of iterations"""
    """Implement function to measure the accuracy"""
    for i in range(NUM_EPOCHS):
        # shuffle
        idx = np.arange(len(train_data))
        np.random.shuffle(idx)
        train_data = train_data[idx]
        train_label = train_label[idx]

        for j in range(0, len(train_data), BSZ):
            Network.Train(
                train_data[j : j + BSZ],
                train_label[j : j + BSZ],
            )
        # Network.UpdateLR()
        o, *_ = Network.Forward(test_data)
        test_acc = np.sum(
            np.argmax(o, axis=1)
            == np.argmax(test_label, axis=1)
        ) / len(test_data)
        if DEBUG:
            print(f"Test Accuracy: {test_acc:.4f}")

    train_acc = Network.Forward(train_data)[0]
    train_acc = np.sum(
        np.argmax(train_acc, axis=1)
        == np.argmax(train_label, axis=1)
    ) / len(train_data)

    print(f"{train_acc:.3f}")
    print(f"{test_acc:.3f}")
    print(f"{NUM_EPOCHS*len(train_data)}")
    print(f"{LR}")


if __name__ == "__main__":
    main()
