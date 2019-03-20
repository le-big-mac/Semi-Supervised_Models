import torch
from torch import nn
from Models.SimpleNetwork import SimpleNetwork
from utils import datautils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def MNIST_train():
    # TODO: change unsupervised argument to be True/False
    _, supervised_dataset, validation_dataset, test_dataset = datautils.load_MNIST_data(100, 100, True, True)

    # run 5 times to get average accuracy
    epochs_list = []
    losses_list = []
    validation_accs_list = []
    results_list = []
    for i in range(5):
        network = SimpleNetwork(784, [1000, 500, 250, 250, 250], 10, nn.ReLU(), device)

        epochs, losses, validation_accs = network.train('MNIST', supervised_dataset, 100, validation_dataset)
        results = network.test(test_dataset)

        epochs_list.append(epochs)
        losses_list.append(losses)
        validation_accs_list.append(validation_accs)
        results_list.append(results)

    datautils.save_results(epochs_list, 'MNIST', 'simple_network', 'epochs')
    datautils.save_results(losses_list, 'MNIST', 'simple_network', 'losses')
    datautils.save_results(validation_accs_list, 'MNIST', 'simple_network', 'validation_accs')
    datautils.save_results(results_list, 'MNIST', 'simple_network', 'test_accuracy')


if __name__ == '__main__':
    MNIST_train()
