from utils.datautils import *
import pickle
import argparse


parser = argparse.ArgumentParser(description='Take arguments to construct model')
parser.add_argument('num_labelled', type=int, help='Number of labelled examples to use')
parser.add_argument('num_folds', type=int, help='Number of folds')
args = parser.parse_args()
num_labelled = args.num_labelled
num_folds = args.num_folds
(train_data, train_labels), _ = load_MNIST_data()

folds = list(stratified_k_fold(train_data, train_labels, num_folds))
label_indices_list = []

for train_index, val_index in folds:
    data = train_data[train_index]
    labels = train_labels[train_index]

    label_indices_list.append(labelled_split(data, labels, num_labelled, True))

folds_and_labels = [folds, label_indices_list]

pickle.dump(folds_and_labels, open('./data/MNIST/{}_labelled_{}_folds.p'.format(num_labelled, num_folds), 'wb'))