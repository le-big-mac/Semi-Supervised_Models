import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets
from collections import defaultdict
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


def stratified_labelled_split(data, labels, num_labelled):
    labels_count = np.unique(labels.numpy(), return_counts=True)
    total_samples = labels.size(0)

    assert (num_labelled > len(labels_count[0]))

    # shuffle tensors
    shuffled = torch.randperm(total_samples)
    data = data[shuffled]
    labels = labels[shuffled]

    label_index_dict = {}

    for lab, count in labels_count:
        relative = ceil((count/total_samples) * num_labelled)
        lab_indexes = (labels == lab).nonzero().squeeze(1)

        label_index_dict[lab] = lab_indexes[:relative]

    labelled_samples_len = sum(tens.size(0) for tens in label_index_dict.values())

    for i in range(labelled_samples_len - num_labelled):
        # remove one from the maximum each time
        max_key = max(label_index_dict, key=lambda x: label_index_dict[x].size(0))
        label_index_dict[max_key] = label_index_dict[max_key][:-1]

    labelled_indices = torch.cat(list(label_index_dict.values()))
    np.random.shuffle(labelled_indices.numpy())
    labelled_data = data[labelled_indices]
    labelled_labels = labels[labelled_indices]

    supervised_dataset = TensorDataset(labelled_data, labelled_labels)
    unsupervised_dataset = TensorDataset(data, -1 * torch.ones(labels.size(0)))

    return supervised_dataset, unsupervised_dataset


class ImputationType(Enum):
    DROP_SAMPLES = 1
    DROP_GENES = 2
    MEAN_VALUE = 3
    ZERO = 4


def load_tcga_data(num_labelled, num_unlabelled, validation=True, test=True,
                   imputation_type=ImputationType.DROP_SAMPLES, stratified_labelled=True):
    # assert num_unlabelled > num_labelled or num_unlabelled == 0

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
        unsupervised_dataset = TensorDataset(normalizer.apply_test(torch.stack(unlabelled_data[0]).float()),
                                             -1 * torch.ones(len(unlabelled_data[1])).long())

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
    assert num_labelled % 10 == 0

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

    shuffled = torch.randperm(train_labels.size(0))
    train_data = train_data[shuffled]
    train_labels = train_labels[shuffled]

    labels_count = np.unique(train_labels.numpy(), return_counts=True)

    labelled_index_dict = {}
    unlabelled_index_dict = {}
    validation_index_dict = {}

    for lab, count in labels_count:
        lab_indexes = (train_labels == lab).nonzero().squeeze(1)

        labelled_index_dict[lab] = lab_indexes[:labelled_per_class]
        unlabelled_index_dict[lab] = lab_indexes[:unlabelled_per_class]
        if validation:
            validation_index_dict[lab] = lab_indexes[unlabelled_per_class:]

    labelled_indices = torch.cat(list(labelled_index_dict.values()))
    # this works because they share the underlying data
    np.random.shuffle(labelled_indices.numpy())
    labelled_data = train_data[labelled_indices]
    labelled_labels = train_labels[labelled_indices]

    supervised_dataset = TensorDataset(labelled_data, labelled_labels)

    unsupervised_dataset = None
    if num_unlabelled > 0:
        unlabelled_indices = torch.cat(list(unlabelled_index_dict.values()))
        np.random.shuffle(unlabelled_indices.numpy())
        unlabelled_data = train_data[unlabelled_indices]

        unsupervised_dataset = TensorDataset(unlabelled_data, -1 * torch.ones(len(unlabelled_indices)))

    validation_dataset = None
    if validation:
        validation_indices = torch.cat(list(validation_index_dict.values()))
        np.random.shuffle(validation_indices.numpy())
        validation_data = train_data[validation_indices]
        validation_labels = train_labels[validation_indices]

        validation_dataset = TensorDataset(validation_data, validation_labels)

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


def other_stratified_split(data, labels, num_labelled):
    labels_np = labels.numpy()
    lab = np.unique(labels_np)
    num_classes = len(lab)

    assert(num_labelled > len(lab))

    total_samples = labels.size(0)

    # shuffle tensors
    shuffled = torch.randperm(total_samples)
    data = data[shuffled]
    labels = labels[shuffled]

    # if randomly shuffled can select first num_labelled
    labelled_data_indices = list(range(num_labelled))

    lab_lab, lab_count = np.unique(labels[:num_labelled].numpy(), return_counts=True)

    # make sure there's at least one of each label
    for i in list(set(lab) - set(lab_lab)):
        # get the first index of the missing class
        index = (labels == i).nonzero()[0].item()

        # find the label with the most values
        max_index = np.argmax(lab_count)
        max_lab = lab_lab[max_index]

        # remove one of the max elements
        first_index_max = (labels == max_lab).nonzero()[0].item()
        labelled_data_indices.remove(first_index_max)
        lab_count[max_index] -= 1

        # add in the indices of the missing class
        labelled_data_indices.append(index)
        lab_lab.append(i)
        lab_count.append(1)

    labelled_data = data[labelled_data_indices]
    labelled_labels = labels[labelled_data_indices]

    assert len(np.unique(labelled_labels.numpy()) == num_classes)

    supervised_dataset = TensorDataset(labelled_data, labelled_labels)
    unsupervised_dataset = TensorDataset(data, -1 * torch.ones(labels.size(0)))

    return supervised_dataset, unsupervised_dataset
