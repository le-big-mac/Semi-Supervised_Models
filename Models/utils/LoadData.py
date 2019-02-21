import numpy as np
from torchvision import datasets
from Models.utils import Datasets


def load_data_from_file(unsupervised_file_path, supervised_data_file_path, supervised_labels_file_path):

    unsupervised_data = np.loadtxt(unsupervised_file_path, dtype=float, delimiter=",")
    supervised_data = np.loadtxt(supervised_data_file_path, dtype=float, delimiter=",")
    supervised_labels = np.loadtxt(supervised_labels_file_path, dtype=int, delimiter=",")

    return unsupervised_data, supervised_data, supervised_labels


def load_MNIST_data(num_labelled, num_validation, num_test, num_unlabelled=0):
    mnist_train = datasets.MNIST(root='../data/MNIST', train=True, download=True, transform=None)
    mnist_test = datasets.MNIST(root='../data/MNIST', train=False, download=True, transform=None)

    unsupervised_dataset = Datasets.MNISTUnsupervised(mnist_train.train_data[:num_unlabelled])

    train_end_index = num_unlabelled + num_labelled

    supervised_dataset = Datasets.MNISTSupervised(mnist_train.train_data[num_unlabelled:train_end_index],
                                                  mnist_train.train_labels[num_unlabelled:train_end_index])

    validation_end_index = train_end_index + num_validation

    validation_dataset = Datasets.MNISTSupervised(mnist_train.train_data[train_end_index:validation_end_index],
                                                  mnist_train.train_labels[train_end_index:validation_end_index])

    test_dataset = Datasets.MNISTSupervised(mnist_test.test_data[:num_test], mnist_test.test_labels[:num_test])

    return unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset
