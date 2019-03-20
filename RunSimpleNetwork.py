import torch
from torch import nn
from Models.SimpleNetwork import SimpleNetwork
from utils import arguments, datautils


def MNIST_train(device):
    # TODO: change unsupervised argument to be True/False
    _, supervised_dataset, validation_dataset, test_dataset = datautils.load_MNIST_data(100, 100, True, True)

    # run 5 times to get average accuracy
    results = []
    for i in range(5):
        network = SimpleNetwork(784, [1000, 500, 250, 250, 250], 10, nn.ReLU(), device)

        print(network.Classifier)

        network.train(supervised_dataset, validation_dataset)

        results.append(network.test(test_dataset))

    datautils.save_results(results, 'simple_network', 'MNIST_accuracy')


def file_train(device):

    args = arguments.parse_args()

    _, supervised_data, supervised_labels = datautils.load_data_from_file(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    simple_network = SimpleNetwork(500, [200], 10, nn.ReLU(), device)

    test_results = []
    for test_idx, train_idx in datautils.k_fold_splits(len(supervised_data), 10):
        train_dataset = datautils.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
                                                                  [supervised_labels[i] for i in train_idx])
        test_dataset = datautils.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
                                                                 [supervised_labels[i] for i in test_idx])

        simple_network.train(train_dataset)

        correct_percentage = simple_network.test(test_dataset)

        test_results.append(correct_percentage)

    datautils.save_results([test_results], 'simple_network', 'accuracy')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MNIST_train(device)
