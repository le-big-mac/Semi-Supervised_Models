import csv


def load_data(unsupervised_file_path, supervised_data_file_path, supervised_labels_file_path):

    unsupervised_data = list(csv.reader(open(unsupervised_file_path)))
    supervised_data = list(csv.reader(open(supervised_data_file_path)))
    supervised_labels = list(csv.reader(open(supervised_labels_file_path)))

    return unsupervised_data, supervised_data, supervised_labels
