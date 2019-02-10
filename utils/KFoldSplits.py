import numpy as np


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
