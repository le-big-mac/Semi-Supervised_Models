import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets
from collections import defaultdict
from sklearn.datasets import make_classification
import pandas as pd
from enum import Enum
from math import ceil


class NormalizeTensors:
    def __init__(self):
        self.means = None
        self.stds = None

    def apply_train(self, data):
        self.means = data.mean(dim=0, keepdim=True)
        self.stds = data.std(dim=0, unbiased=False, keepdim=True)

        norm_data = (data - self.means) / (1e-7 + self.stds)

        return norm_data

    def apply_test(self, data):
        norm_data = (data - self.means) / (1e-7 + self.stds)

        return norm_data


def make_toy_data(n_samples, n_features, n_classes):
    data, labels = make_classification(n_samples=n_samples, n_features=n_features, n_informative=10, n_redundant=20,
                                       n_classes=n_classes, n_clusters_per_class=1, shuffle=False)

    dataset = zip(data, labels)

    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./data/toy'):
        os.mkdir('./data/toy')

    toy_data_file = open('./data/toy/toy_data.csv', 'w')
    toy_labels_file = open('./data/toy/toy_labels.csv', 'w')

    data_writer = csv.writer(toy_data_file)
    labels_writer = csv.writer(toy_labels_file)

    for d, l in dataset:
        data_writer.writerow(d)
        labels_writer.writerow([l])


def load_toy_data(num_labelled, num_unlabelled=0, validation=True, test=True):
    assert num_unlabelled > num_labelled or num_unlabelled == 0

    if not os.path.exists('./data/toy/toy_data.csv'):
        num_samples = max(num_labelled, num_unlabelled) + int(validation) * 1000 + int(test) * 1000

        make_toy_data(num_samples, 500, 4)

    data = np.loadtxt('./data/toy/toy_data.csv', dtype=float, delimiter=',')
    labels = np.loadtxt('./data/toy/toy_labels.csv', dtype=int, delimiter=',')

    input_size = len(data[0])
    num_classes = len(set(labels))

    labelled_per_class = num_labelled // num_classes
    unlabelled_per_class = num_unlabelled // num_classes

    data_buckets = defaultdict(list)

    for d, l in zip(data, labels):
        data_buckets[l].append((d, l))

    unlabelled_data = []
    labelled_data = []
    validation_data = []
    test_data = []

    for l, d in data_buckets.items():
        labelled_data.extend(d[:labelled_per_class])
        unlabelled_data.extend(d[:unlabelled_per_class])

        lower = max(labelled_per_class, unlabelled_per_class)
        if validation and test:
            leftover = len(d) - lower
            validation_data.extend(d[lower:lower + leftover // 2])
            test_data.extend(d[lower + leftover // 2:])
        elif validation:
            validation_data.extend(d[lower:])
        elif test:
            test_data.extend(d[lower:])

    np.random.shuffle(labelled_data)
    np.random.shuffle(unlabelled_data)
    np.random.shuffle(validation_data)
    np.random.shuffle(test_data)

    labelled_data = list(zip(*labelled_data))
    unlabelled_data = list(zip(*unlabelled_data))
    validation_data = list(zip(*validation_data))
    test_data = list(zip(*test_data))

    supervised_dataset = TensorDataset(torch.from_numpy(np.stack(labelled_data[0])).float(),
                                       torch.from_numpy(np.array(labelled_data[1])).long())

    unsupervised_dataset = None
    if num_unlabelled > 0:
        unsupervised_dataset = TensorDataset(torch.from_numpy(np.stack(unlabelled_data[0])).float(),
                                             torch.from_numpy(np.array([-1] * len(unlabelled_data[1]))).long())
    validation_dataset = None
    if validation:
        validation_dataset = TensorDataset(torch.from_numpy(np.stack(validation_data[0])).float(),
                                           torch.from_numpy(np.array(validation_data[1])).long())
    test_dataset = None
    if test:
        test_dataset = TensorDataset(torch.from_numpy(np.stack(test_data[0])).float(),
                                     torch.from_numpy(np.array(test_data[1])).long())

    return (unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset), input_size, num_classes


class ImputationType(Enum):
    DROP_SAMPLES = 1
    DROP_GENES = 2
    MEAN_VALUE = 3
    ZERO = 4


def load_tcga_data(num_labelled, num_unlabelled, validation=True, test=True,
                   imputation_type=ImputationType.DROP_SAMPLES, stratified_labelled=True):
    assert num_unlabelled > num_labelled or num_unlabelled == 0

    rnaseq_df = pd.read_csv('data/tcga/rnaseq_data_with_labels.csv', index_col=0)

    if imputation_type == ImputationType.DROP_SAMPLES:
        rnaseq_df = rnaseq_df.dropna(axis=0, how='any')
    elif imputation_type == ImputationType.DROP_GENES:
        rnaseq_df = rnaseq_df.dropna(axis=1, how='any')
    elif imputation_type == ImputationType.MEAN_VALUE:
        col_means = rnaseq_df.mean()
        rnaseq_df = rnaseq_df.fillna(col_means)
    else:
        rnaseq_df = rnaseq_df.fillna(0)

    num_samples_map = rnaseq_df['DISEASE'].value_counts().to_dict()
    total_samples = len(rnaseq_df.index)
    num_classes = len(num_samples_map)
    input_size = len(rnaseq_df.columns) - 1

    disease_label_map = {d: i for i, d in enumerate(list(num_samples_map.keys()))}

    labelled_data = []
    unlabelled_data = []
    validation_data = []
    test_data = []

    labelled_per_sample = num_labelled // num_classes
    unlabelled_per_sample = num_unlabelled // num_classes

    for disease, num_samples in num_samples_map.items():
        sample_rows = rnaseq_df.loc[rnaseq_df['DISEASE'] == disease]
        sample_rows = sample_rows.drop('DISEASE', axis=1)
        disease_tensor = torch.tensor(sample_rows.values)
        labels_tensor = disease_label_map[disease] * torch.ones(disease_tensor.size(0))
        data_list = list(zip(disease_tensor, labels_tensor))

        if stratified_labelled:
            relative = num_samples/total_samples
            labelled_per_sample = int(ceil(num_labelled * relative))
            unlabelled_per_sample = int(ceil(num_unlabelled * relative))

        labelled_data.extend(data_list[:labelled_per_sample])
        unlabelled_data.extend(data_list[:unlabelled_per_sample])

        lower = max(labelled_per_sample, unlabelled_per_sample)
        if validation and test:
            leftover = len(data_list) - lower
            validation_data.extend(data_list[lower:lower + leftover // 2])
            test_data.extend(data_list[lower + leftover // 2:])
        elif validation:
            validation_data.extend(data_list[lower:])
        elif test:
            test_data.extend(data_list[lower:])

    np.random.shuffle(labelled_data)
    np.random.shuffle(unlabelled_data)
    np.random.shuffle(validation_data)
    np.random.shuffle(test_data)

    labelled_data = list(zip(*labelled_data))
    unlabelled_data = list(zip(*unlabelled_data))
    validation_data = list(zip(*validation_data))
    test_data = list(zip(*test_data))

    normalizer = NormalizeTensors()

    unsupervised_dataset = None
    if num_unlabelled > num_labelled:
        unsupervised_dataset = TensorDataset(normalizer.apply_train(torch.stack(unlabelled_data[0]).float()),
                                             -1 * torch.ones(len(unlabelled_data[1])).long())
        supervised_dataset = TensorDataset(normalizer.apply_test(torch.stack(labelled_data[0]).float()),
                                           torch.stack(labelled_data[1]).long())
    else:
        supervised_dataset = TensorDataset(normalizer.apply_train(torch.stack(labelled_data[0]).float()),
                                           torch.stack(labelled_data[1]).long())

    validation_dataset = None
    if validation:
        validation_dataset = TensorDataset(normalizer.apply_test(torch.stack(validation_data[0]).float()),
                                           torch.stack(validation_data[1]).long())
    test_dataset = None
    if test:
        test_dataset = TensorDataset(normalizer.apply_test(torch.stack(test_data[0]).float()),
                                     torch.stack(test_data[1]).long())

    return (unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset), input_size, num_classes


def load_MNIST_data(num_labelled, num_unlabelled=0, validation=True, test=True):
    assert num_unlabelled > num_labelled or num_unlabelled == 0

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

    supervised_dataset = TensorDataset(torch.stack(labelled_data[0]), torch.stack(labelled_data[1]))

    unsupervised_dataset = None
    if num_unlabelled > 0:
        unsupervised_dataset = TensorDataset(torch.stack(unlabelled_data[0]),
                                             -1 * torch.ones(len(unlabelled_data[1])).long())

    validation_dataset = None
    if validation:
        validation_dataset = TensorDataset(torch.stack(validation_data[0]), torch.stack(validation_data[1]))

    test_dataset = None
    if test:
        test_dataset = TensorDataset(test_data, test_labels)

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
