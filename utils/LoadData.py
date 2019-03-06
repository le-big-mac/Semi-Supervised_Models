import torch
import numpy as np
from torchvision import datasets
from utils import Datasets
from collections import defaultdict


def load_data_from_file(unsupervised_file_path, supervised_data_file_path, supervised_labels_file_path):

    unsupervised_data = np.loadtxt(unsupervised_file_path, dtype=float, delimiter=",")
    supervised_data = np.loadtxt(supervised_data_file_path, dtype=float, delimiter=",")
    supervised_labels = np.loadtxt(supervised_labels_file_path, dtype=int, delimiter=",")

    return unsupervised_data, supervised_data, supervised_labels


def load_MNIST_data(num_labelled, num_unlabelled=0, validation=True, test=True):
    mnist_train = datasets.MNIST(root='data/MNIST', train=True, download=True, transform=None)
    mnist_test = datasets.MNIST(root='data/MNIST', train=False, download=True, transform=None)

    train_data = mnist_train.train_data
    train_labels = mnist_train.train_labels

    test_data = mnist_test.test_data
    test_labels = mnist_test.test_labels

    train_data = train_data.view(-1, 784)
    train_data = 1./255. * train_data.double()
    test_data = test_data.view(-1, 784)
    test_data = 1./255. * test_data.double()

    labelled_per_class = num_labelled//10
    unlabelled_per_class = num_unlabelled//10

    buckets_train_data = defaultdict(list)

    for image, label in zip(train_data, train_labels):
        buckets_train_data[label.item()].append((image, label))

    labelled_data = []
    unlabelled_data = []
    validation_data = []

    for label, data in buckets_train_data.items():
        labelled_data.extend(data[:labelled_per_class])
        unlabelled_data.extend(data[labelled_per_class:labelled_per_class + unlabelled_per_class])
        if validation:
            validation_data.extend(data[labelled_per_class + unlabelled_per_class:])

    np.random.shuffle(labelled_data)
    np.random.shuffle(unlabelled_data)
    np.random.shuffle(validation_data)

    labelled_data = list(zip(*labelled_data))
    unlabelled_data = list(zip(*unlabelled_data))
    validation_data = list(zip(*validation_data))

    supervised_dataset = Datasets.MNISTSupervised(torch.stack(labelled_data[0]), torch.stack(labelled_data[1]))
    unsupervised_dataset = Datasets.MNISTUnsupervised(torch.stack(unlabelled_data[0]))

    validation_dataset = None
    if validation:
        validation_dataset = Datasets.MNISTSupervised(torch.stack(validation_data[0]), torch.stack(validation_data[1]))

    test_dataset = None
    if test:
        test_dataset = Datasets.MNISTSupervised(test_data, test_labels)

    return unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset
