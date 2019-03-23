import torch
from torch import nn
from Models.PretrainingNetwork_new import PretrainingNetwork
from torch.utils.data import DataLoader
from utils import arguments, datautils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def MNIST_train():
    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
        datautils.load_MNIST_data(100, 50000, True, True)

    results = []
    for i in range(5):
        deep_metabolism = PretrainingNetwork(784, [1000, 500, 250, 250, 250], 10, lambda x: x, nn.Sigmoid(), device)

        print(deep_metabolism.Autoencoder)

        supervised_dataloader = DataLoader(supervised_dataset, batch_size=100, shuffle=True)
        unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=100, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=validation_dataset.__len__())

        deep_metabolism.train('a', supervised_dataloader, unsupervised_dataloader, validation_dataloader)

        results.append(deep_metabolism.test(test_dataset))

    datautils.save_results(results, 'MNIST_debug', 'pretraining_new', 'results')


if __name__ == '__main__':
    MNIST_train()
