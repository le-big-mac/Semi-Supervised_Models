import torch
from torch import nn
from Models import M1
from utils import datautils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def MNIST_train():

    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
        datautils.load_MNIST_data(100, 49900, True, True)

    combined_dataset = datautils.MNISTUnsupervised(torch.cat((unsupervised_dataset.data, supervised_dataset.data), 0))

    results = []
    for i in range(5):
        m1 = M1(784, [256, 128], 32, [32], 10, nn.Sigmoid(), device)

        print(m1.VAE)
        print(m1.Classifier)

        m1.train(combined_dataset, supervised_dataset, validation_dataset)

        results.append(m1.test(test_dataset))

    datautils.save_results(results, 'm1', 'MNIST_accuracy')


# def file_train():
#
#     args = arguments.parse_args()
#
#     unsupervised_data, supervised_data, supervised_labels = datautils.load_data_from_file(
#         args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     m1 = M1(500, [200], 10, nn.ReLU(), device)
#
#     test_results = []
#     for test_idx, train_idx in datautils.k_fold_splits(len(supervised_data), 10):
#         train_dataset = datautils.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
#                                                                   [supervised_labels[i] for i in train_idx])
#         test_dataset = datautils.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
#                                                                  [supervised_labels[i] for i in test_idx])
#
#         m1.full_train(train_dataset)
#
#         correct_percentage = m1.full_test(test_dataset)
#
#         test_results.append(correct_percentage)
#
#     datautils.save_results([test_results], 'm1', 'accuracy')


if __name__ == '__main__':
    MNIST_train()
