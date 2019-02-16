import numpy as np


def load_data(unsupervised_file_path, supervised_data_file_path, supervised_labels_file_path):

    unsupervised_data = np.loadtxt(unsupervised_file_path, dtype=float, delimiter=",")
    supervised_data = np.loadtxt(supervised_data_file_path, dtype=float, delimiter=",")
    supervised_labels = np.loadtxt(supervised_labels_file_path, dtype=int, delimiter=",")

    return unsupervised_data, supervised_data, supervised_labels
