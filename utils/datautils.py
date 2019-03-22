import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from collections import defaultdict


class SupervisedClassificationDataset(Dataset):

    def __init__(self, inputs, outputs):
        super(SupervisedClassificationDataset, self).__init__()

        self.data = [torch.from_numpy(np.atleast_1d(vec)).float() for vec in inputs]
        self.labels = [torch.squeeze(torch.from_numpy(np.atleast_1d(vec)).long()) for vec in outputs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class UnsupervisedDataset(Dataset):

    def __init__(self, inputs):
        super(UnsupervisedDataset, self).__init__()

        self.data = [torch.from_numpy(np.atleast_1d(vec)).float() for vec in inputs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class MNISTSupervised(Dataset):
    def __init__(self, data, labels):

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        return img, label


class MNISTUnsupervised(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data_from_file(unsupervised_file_path, supervised_data_file_path, supervised_labels_file_path):

    unsupervised_data = np.loadtxt(unsupervised_file_path, dtype=float, delimiter=",")
    supervised_data = np.loadtxt(supervised_data_file_path, dtype=float, delimiter=",")
    supervised_labels = np.loadtxt(supervised_labels_file_path, dtype=int, delimiter=",")

    return unsupervised_data, supervised_data, supervised_labels


def load_MNIST_data(num_labelled, num_unlabelled=0, validation=True, test=True):
    mnist_train = datasets.MNIST(root='data/MNIST', train=True, download=True, transform=None)
    mnist_test = datasets.MNIST(root='data/MNIST', train=False, download=True, transform=None)

    train_data = mnist_train.data
    train_labels = mnist_train.targets

    test_data = mnist_test.data
    test_labels = mnist_test.targets

    train_data = train_data.view(-1, 784)
    train_data = 1./255. * train_data.float()
    test_data = test_data.view(-1, 784)
    test_data = 1./255. * test_data.float()

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
        # changed so that unlabelled data contains the labelled data - to do with cycling supervised data pairs
        unlabelled_data.extend(data[:unlabelled_per_class])
        if validation:
            validation_data.extend(data[unlabelled_per_class:])

    np.random.shuffle(labelled_data)
    np.random.shuffle(unlabelled_data)
    np.random.shuffle(validation_data)

    labelled_data = list(zip(*labelled_data))
    unlabelled_data = list(zip(*unlabelled_data))
    validation_data = list(zip(*validation_data))

    supervised_dataset = MNISTSupervised(torch.stack(labelled_data[0]), torch.stack(labelled_data[1]))
    unsupervised_dataset = MNISTUnsupervised(torch.stack(unlabelled_data[0]))

    validation_dataset = None
    if validation:
        validation_dataset = MNISTSupervised(torch.stack(validation_data[0]), torch.stack(validation_data[1]))

    test_dataset = None
    if test:
        test_dataset = MNISTSupervised(test_data, test_labels)

    return unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset


def save_results(results_list, dataset_directory, model_directory, filename):
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/{}'.format(dataset_directory)):
        os.mkdir('results/{}'.format(dataset_directory))
    if not os.path.exists('results/{}/{}'.format(dataset_directory, model_directory)):
        os.mkdir('results/{}/{}'.format(dataset_directory, model_directory))

    # if os.path.exists('results/{}/{}/{}.csv'.format(dataset_directory, model_directory, filename)):
    #     os.remove('results/{}/{}/{}.csv'.format(dataset_directory, model_directory, filename))

    file = open('results/{}/{}/{}.csv'.format(dataset_directory, model_directory, filename), 'w')
    writer = csv.writer(file)

    if not isinstance(results_list, list):
        raise ValueError

    if isinstance(results_list[0], list):
        for row in results_list:
            writer.writerow(row)
    else:
        writer.writerow(results_list)

    file.close()


def k_fold_splits(len_data, num_folds):
    indices = list(range(len_data))

    test_idx = []
    train_idx = []

    for index_chunk in np.array_split(indices, num_folds):
        test_idx.append(index_chunk)
        train_idx.append(list(set(indices) - set(index_chunk)))

    return zip(test_idx, train_idx)


def k_fold_splits_with_validation(len_data, num_folds):
    indices = list(range(len_data))

    test_idx = []
    validation_idx = []
    train_idx = []

    for index_chunk in np.array_split(indices, num_folds):
        test_idx.append(index_chunk[:len(index_chunk)//2])
        validation_idx.append(index_chunk[len(index_chunk)//2:])
        train_idx.append(list(set(indices) - set(index_chunk)))

    return zip(test_idx, validation_idx, train_idx)
