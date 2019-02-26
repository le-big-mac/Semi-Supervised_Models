import torch
from torch import nn
from Models.Pretraining.DeepMetabolism import DeepMetabolism
from utils import LoadData, Datasets, Arguments, KFoldSplits, SaveResults


def MNIST_train(device):
    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
        LoadData.load_MNIST_data(100, 10000, 10000, 49900)

    combined_dataset = Datasets.MNISTUnsupervised(torch.cat((unsupervised_dataset.data, supervised_dataset.data), 0))

    results = []
    for i in range(5):
        deep_metabolism = DeepMetabolism(784, [1000, 500, 250, 250, 250], 10, nn.ReLU(), device)

        print(deep_metabolism.Classifier)

        deep_metabolism.full_train(combined_dataset, supervised_dataset, validation_dataset)

        results.append(deep_metabolism.full_test(test_dataset))

    SaveResults.save_results(results, 'deep_metabolism', 'MNIST_accuracy')


def file_train(device):

    # TODO: this is all wrong
    args = Arguments.parse_args()

    unsupervised_data, supervised_data, supervised_labels = LoadData.load_data_from_file(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    deep_metabolism = DeepMetabolism(500, [200], 10, nn.ReLU(), device)

    test_results = []
    for test_idx, train_idx in KFoldSplits.k_fold_splits(len(supervised_data), 10):
        train_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
                                                                 [supervised_labels[i] for i in train_idx])
        test_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
                                                                [supervised_labels[i] for i in test_idx])

        deep_metabolism.full_train(train_dataset)

        correct_percentage = deep_metabolism.full_test(test_dataset)

        test_results.append(correct_percentage)

    SaveResults.save_results([test_results], 'deep_metabolism', 'blah')


if __name__ == '__main__':

    MNIST_train('cpu')
