import torch
from torch import nn
from Models import M2Runner
from utils import datautils


def MNIST_train(device):

    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
        datautils.load_MNIST_data(100, num_unlabelled=49900, validation=True, test=True)

    alpha = 0.1 * 49900/100

    results = []
    for i in range(5):
        m2 = M2Runner(784, [256, 128], [256], 32, 10, nn.Sigmoid(), device)

        print(m2.M2)

        m2.train(unsupervised_dataset, supervised_dataset, validation_dataset, alpha)

        results.append(m2.test(test_dataset))

    datautils.save_results(results, 'm2', 'MNIST_accuracy')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MNIST_train(device)
