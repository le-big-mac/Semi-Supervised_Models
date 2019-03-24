import torch
from torch import nn
from Models.PretrainingNetwork_old import PretrainingNetwork
from utils import datautils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def MNIST_train():
    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
        datautils.load_MNIST_data(100, 50000, True, True)

    results = []
    for i in range(5):
        deep_metabolism = PretrainingNetwork(784, [1000, 500, 250, 250, 250], 10, lambda x: x, nn.Sigmoid(), device)

        print(deep_metabolism.Autoencoder)

        deep_metabolism.train(unsupervised_dataset, supervised_dataset, validation_dataset)

        results.append(deep_metabolism.test(test_dataset))

    datautils.save_results(results, 'MNIST_debug', 'pretraining', 'results')


if __name__ == '__main__':
    MNIST_train()
