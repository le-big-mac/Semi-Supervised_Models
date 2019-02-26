import torch
from torch import nn
from Models.M1 import M1
from utils import Arguments, LoadData, Datasets, KFoldSplits, SaveResults


def MNIST_train(device):

    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
        LoadData.load_MNIST_data(100, 10000, 10000, 49000)

    combined_dataset = Datasets.MNISTUnsupervised(torch.cat((unsupervised_dataset.data, supervised_dataset.data), 0))

    results = []
    for i in range(5):
        m1 = M1(784, [1000, 500, 250], 250, [250], 10, nn.ReLU(), device)

        print(m1.VAE)
        print(m1.Classifier)

        m1.full_train(combined_dataset, supervised_dataset, validation_dataset)

        results.append(m1.full_test(test_dataset))

    SaveResults.save_results(results, 'm1', 'MNIST_accuracy')


def file_train(device):

    args = Arguments.parse_args()

    unsupervised_data, supervised_data, supervised_labels = LoadData.load_data_from_file(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    m1 = M1(500, [200], 10, nn.ReLU(), device)

    test_results = []
    for test_idx, train_idx in KFoldSplits.k_fold_splits(len(supervised_data), 10):
        train_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
                                                                 [supervised_labels[i] for i in train_idx])
        test_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
                                                                [supervised_labels[i] for i in test_idx])

        m1.full_train(train_dataset)

        correct_percentage = m1.full_test(test_dataset)

        test_results.append(correct_percentage)

    SaveResults.save_results([test_results], 'm1', 'accuracy')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MNIST_train(device)
