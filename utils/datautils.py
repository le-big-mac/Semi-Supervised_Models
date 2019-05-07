import os
import csv
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torchvision import datasets
import pandas as pd
from enum import Enum
from math import ceil
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def stratified_k_fold(data, labels, num_folds=5):
    skf = StratifiedKFold(num_folds)

    return skf.split(data, labels)


def labelled_split(data, labels, num_labelled, stratified=True):
    unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
    total_samples = labels.size(0)
    num_classes = len(unique_labels)

    assert(num_labelled > num_classes)

    label_index_dict = {}

    # unstratified won't return as many labelled as expected if labelled_per_class is bigger than smallest class
    labelled_per_class = ceil(num_labelled/num_classes)
    for lab, count in zip(unique_labels, counts):
        relative = ceil((count/total_samples) * num_labelled) if stratified else labelled_per_class
        lab_indexes = (labels == lab).nonzero().squeeze(1)

        label_index_dict[lab] = lab_indexes[:relative]

    labelled_samples_len = sum(tens.size(0) for tens in label_index_dict.values())

    for i in range(labelled_samples_len - num_labelled):
        # remove one from the maximum each time
        max_key = max(label_index_dict, key=lambda x: label_index_dict[x].size(0))
        label_index_dict[max_key] = label_index_dict[max_key][:-1]

    labelled_indices = torch.cat(list(label_index_dict.values()))
    np.random.shuffle(labelled_indices.numpy())

    return labelled_indices


class ImputationType(Enum):
    DROP_SAMPLES = 1
    DROP_GENES = 2
    MEAN_VALUE = 3
    ZERO = 4


def load_train_data_from_file(filepath):
    df = pd.read_csv(filepath, index_col=0, low_memory=False)

    label_column = df[df.columns[-1]]

    data_to_impute = df[df.columns[:-1]]
    col_means = data_to_impute.mean()
    data_to_impute = data_to_impute.fillna(col_means)

    unlabel_mask = label_column.isna()
    labelled_data = data_to_impute[~unlabel_mask]
    labels = label_column[~unlabel_mask]
    unlabelled_data = data_to_impute[unlabel_mask]

    unique_labels = labels.unique()
    string_int_label_map = dict(zip(unique_labels, range(len(unique_labels))))

    labelled_data = torch.tensor(labelled_data.values).float()
    labels = torch.tensor([string_int_label_map[d] for d in labels.values]).long()
    unlabelled_data = torch.tensor(unlabelled_data.values).float()

    int_string_map = {v: k for k, v in string_int_label_map.items()}

    return (labelled_data, labels), unlabelled_data, int_string_map, col_means


def load_data_to_classify_from_file(filepath, col_means):
    df = pd.read_csv(filepath, index_col=0)
    df = df.fillna(col_means)
    sample_names = df.index

    data = torch.tensor(df.values).float()

    return sample_names, data


def load_tcga_data(imputation_type=ImputationType.DROP_SAMPLES):
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

    label_count_map = dict(rnaseq_df['DISEASE'].value_counts())
    remove_labels = [k for k, v in label_count_map.items() if v < 50]

    rnaseq_df = rnaseq_df[~rnaseq_df['DISEASE'].isin(remove_labels)]

    unique_labels = rnaseq_df['DISEASE'].unique()
    string_int_label_map = dict(zip(unique_labels, range(len(unique_labels))))

    labels = torch.tensor([string_int_label_map[d] for d in rnaseq_df['DISEASE'].values]).long()
    data = torch.tensor(rnaseq_df.loc[rnaseq_df.index].drop('DISEASE', axis=1).values).float()

    # rand = torch.randperm(labels.size(0))
    # labels = labels[rand]
    # data = data[rand]

    num_classes = len(unique_labels)
    input_size = data.size(1)

    return (data, labels), (input_size, num_classes)


def load_MNIST_data():
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

    return (train_data, train_labels), (test_data, test_labels)
