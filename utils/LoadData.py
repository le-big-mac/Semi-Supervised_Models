import torch
import numpy as np
from torchvision import datasets
from utils import Datasets


def load_data_from_file(unsupervised_file_path, supervised_data_file_path, supervised_labels_file_path):

    unsupervised_data = np.loadtxt(unsupervised_file_path, dtype=float, delimiter=",")
    supervised_data = np.loadtxt(supervised_data_file_path, dtype=float, delimiter=",")
    supervised_labels = np.loadtxt(supervised_labels_file_path, dtype=int, delimiter=",")

    return unsupervised_data, supervised_data, supervised_labels


def load_MNIST_data(num_labelled, num_validation, num_test, num_unlabelled=0):
    mnist_train = datasets.MNIST(root='data/MNIST', train=True, download=True, transform=None)
    mnist_test = datasets.MNIST(root='data/MNIST', train=False, download=True, transform=None)

    elements_of_each = num_labelled//10

    label_nums = [elements_of_each] * 10

    np_labels = []
    np_labelled_data = []
    np_unlabelled_data = []

    num_labelled = labels = elements_of_each*10
    i = 0
    while labels > 0:
        data = mnist_train.train_data[i]
        label = mnist_train.train_labels[i]

        if label_nums[label.item()] > 0:
            np_labelled_data.append(data)
            np_labels.append(label)

            label_nums[label.item()] -= 1
            labels -= 1

        elif num_unlabelled > len(np_unlabelled_data):
            np_unlabelled_data.append(data)

        i += 1

    final_idx = i

    supervised_dataset = Datasets.MNISTSupervised(torch.stack(np_labelled_data), torch.stack(np_labels))

    unlabelled_tensor = torch.stack(np_unlabelled_data) if num_unlabelled > 0 else None
    if num_unlabelled > len(np_unlabelled_data):
        unlabelled_end_idx = num_unlabelled + num_labelled

        unsupervised_dataset = Datasets.MNISTUnsupervised(
            torch.cat((unlabelled_tensor, mnist_train.train_data[final_idx:unlabelled_end_idx])))

    else:
        unlabelled_end_idx = final_idx
        unsupervised_dataset = Datasets.MNISTUnsupervised(unlabelled_tensor)

    validation_end_index = unlabelled_end_idx + num_validation \
        if unlabelled_end_idx + num_validation < len(mnist_train.train_data) else len(mnist_train.train_data)

    validation_dataset = Datasets.MNISTSupervised(mnist_train.train_data[unlabelled_end_idx:validation_end_index],
                                                  mnist_train.train_labels[unlabelled_end_idx:validation_end_index])

    test_dataset = Datasets.MNISTSupervised(mnist_test.test_data[:num_test], mnist_test.test_labels[:num_test])

    return unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset


if __name__ == '__main__':
    load_MNIST_data(100, 10000, 10000, 49900)
