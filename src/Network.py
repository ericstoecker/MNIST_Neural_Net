"""
    Neural Network to recognize MNIST digits

    example code to train a model with three layers:
        from src import Network
        from src import mnist_loader

        from src.train_network import train_network

        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        option 1:
            net = Network.Network([784, 30, 10])

            net.train(training_data, 30, 10, 3.0, test_data=test_data)

        option 2:
            train_network(Network, layers=[784, 20, 10],  training_data=training_data, epochs=3,
            mini_batch_size=10, eta=[10, 1, 0.1], test_data=test_data)

"""


import numpy as np
import random

from src.train_network import train_network
from src import mnist_loader


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # used to train net
    # returns test cost for each epoch if test_data was given
    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        # list to store percentage of correctly classified Images after each epoch
        correctly_classified = []
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.train_batch(mini_batch, eta)
            if test_data:
                classified = self.evaluate(test_data)
                print("Epoch {0}: {1} / {2}".format(
                    j, classified, n_test))
                correctly_classified.append(classified / n_test)
            else:
                print("Epoch {0} complete".format(j))
        return correctly_classified

    def train_batch(self, batch, eta):
        x = batch[0][0]
        y = batch[0][1]

        nabla_b, nabla_w = self.backpropagation(x, y)
        self.weights = [w - np.multiply(eta, nw)
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - np.multiply(eta, nb)
                       for b, nb in zip(self.biases, nabla_b)]

    def forward_propagation(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backpropagation(self, x, y):
        """ computes gradients of weights and biases as matrix that can be used
        to update the layers parameters """
        # lists to store gradients, similar to self.weights, self.biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # list to store the activations and z-vectors of each layer
        # z = wx + b, activation = f(z)
        activation = x
        activations = [x]
        z_vec = []

        # forward pass
        for w, b in zip(self.weights, self.biases):
            z = np.matmul(w, activation) + b
            z_vec.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        g = self.cost_derivative(activations[-1], y)
        for k in reversed(range(0, self.num_layers - 1)):
            g = np.multiply(g, sigmoid_prime(z_vec[k]))

            nabla_b[k] = g
            nabla_w[k] = np.matmul(g, np.transpose(activations[k]))

            g = np.matmul(np.transpose(self.weights[k]), g)

        return nabla_b, nabla_w

    def cost(self, output_activations, y):
        return

    def cost_derivative(self, output_activations, y):
        m = len(y)
        return (output_activations - y) / m

    # returns the number of correctly classified images
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forward_propagation(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(np.negative(z)))


def sigmoid_prime(z):
    return np.multiply(sigmoid(z), (1.0 - sigmoid(z)))


