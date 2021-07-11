import matplotlib.pyplot as plt


def train_network(network, layers, training_data, epochs, mini_batch_size, eta, test_data=None):
    network = network.Network(layers)

    for i in range(len(eta)):
        print("Eta: {0}".format(eta[i]))
        correctly_classified = network.train(training_data, epochs, mini_batch_size, eta[i], test_data)
        plt.plot(list(range(len(correctly_classified))), correctly_classified, label="Eta: {0}".format(eta[i]))

    plt.ylabel("correctly classified")
    plt.xlabel("epoch")
    plt.legend(loc='best')
    plt.show()
