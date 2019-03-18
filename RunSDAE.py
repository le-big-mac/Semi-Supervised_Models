import torch
from torch import nn
from Models.SDAE import SDAENetwork
from utils import arguments, datautils


def MNIST_train(device):

    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
        datautils.load_MNIST_data(100, 10000, 10000, 49000)

    combined_dataset = datautils.MNISTUnsupervised(torch.cat((unsupervised_dataset.data, supervised_dataset.data), 0))

    results = []
    for i in range(5):
        sdae = SDAENetwork(784, [1000, 500, 250, 250, 250], 10, nn.ReLU(), device)

        print(sdae.SDAE)

        sdae.full_train(combined_dataset, supervised_dataset, validation_dataset)

        results.append(sdae.full_test(test_dataset))

    datautils.save_results(results, 'sdae', 'MNIST_accuracy')


def file_train(device):

    args = arguments.parse_args()

    unsupervised_data, supervised_data, supervised_labels = datautils.load_data_from_file(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sdae = SDAENetwork(500, [200], 10, nn.ReLU(), device)

    test_results = []
    for test_idx, train_idx in datautils.k_fold_splits(len(supervised_data), 10):
        train_dataset = datautils.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
                                                                  [supervised_labels[i] for i in train_idx])
        test_dataset = datautils.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
                                                                 [supervised_labels[i] for i in test_idx])

        sdae.full_train(train_dataset)

        correct_percentage = sdae.full_test(test_dataset)

        test_results.append(correct_percentage)

    datautils.save_results([test_results], 'sdae', 'accuracy')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MNIST_train(device)
