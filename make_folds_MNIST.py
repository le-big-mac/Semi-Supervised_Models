from utils.datautils import *
import sys
import pickle

num_labelled = int(sys.argv[1])
num_folds = int(sys.argv[2])
(train_data, train_labels), (test_data, test_labels) = load_MNIST_data()

folds = list(stratified_k_fold(train_data, train_labels, num_folds))
label_indices_list = []

for train_index, test_index in folds:
    data = train_data[train_index]
    labels = train_labels[train_index]

    label_indices_list.append(labelled_split(data, labels, num_labelled, True))

folds_and_labels = [folds, label_indices_list]

pickle.dump(folds_and_labels, open('./data/MNIST/{}_labelled_{}_folds.p'.format(num_labelled, num_folds), 'wb'))