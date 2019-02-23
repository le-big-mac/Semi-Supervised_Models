import torch
from torch import nn
from Models.SimpleNetwork import SimpleNetwork
from utils import LoadData, Datasets, Arguments, KFoldSplits, SaveResults


def MNIST_train(device):

    _, supervised_dataset, validation_dataset, test_dataset = LoadData.load_MNIST_data(100, 10000, 10000, 0)

    network = SimpleNetwork(784, [1000, 500, 250, 250, 250], 10, nn.ReLU(), device)

    print(network.Classifier)

    network.full_train(supervised_dataset, validation_dataset)

    return network.full_test(test_dataset)


def file_train(device):

    args = Arguments.parse_args()

    _, supervised_data, supervised_labels = LoadData.load_data_from_file(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    simple_network = SimpleNetwork(500, [200], 10, nn.ReLU(), device)

    test_results = []
    for test_idx, train_idx in KFoldSplits.k_fold_splits(len(supervised_data), 10):
        train_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
                                                                 [supervised_labels[i] for i in train_idx])
        test_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
                                                                [supervised_labels[i] for i in test_idx])

        simple_network.full_train(train_dataset)

        correct_percentage = simple_network.full_test(test_dataset)

        test_results.append(correct_percentage)

    SaveResults.save_results([test_results], 'simple_network')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MNIST_train(device)
