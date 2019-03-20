import torch
from torch import nn
from Models import SimpleM1
from utils import datautils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def MNIST_train():

    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
        datautils.load_MNIST_data(100, 49900, True, True)

    combined_dataset = datautils.MNISTUnsupervised(torch.cat((unsupervised_dataset.data, supervised_dataset.data), 0))

    results = []
    for i in range(5):
        m1 = SimpleM1(784, [256, 128], 32, [32], 10, lambda x: x, nn.Sigmoid(), device)

        print(m1.Autoencoder)
        print(m1.Classifier)

        m1.train(combined_dataset, supervised_dataset, validation_dataset)

        results.append(m1.test(test_dataset))

    datautils.save_results(results, 'm1', 'MNIST_accuracy')


if __name__ == '__main__':
    MNIST_train()
