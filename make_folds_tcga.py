from utils.datautils import *
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Take arguments to construct model')
parser.add_argument('num_labelled', type=int, help='Number of labelled examples to use')
parser.add_argument('num_folds', type=int, help='Number of folds')
parser.add_argument('--drop_samples', default=False, action='store_true', help='Drop samples')
args = parser.parse_args()
drop_samples = args.drop_samples
print(drop_samples)
num_labelled = args.num_labelled
num_folds = args.num_folds
(data, labels), (input_size, num_classes) = load_tcga_data(ImputationType.DROP_SAMPLES) if drop_samples else \
    load_tcga_data(ImputationType.DROP_GENES)

train_test_folds = list(stratified_k_fold(data, labels, num_folds))
label_indices_list = []
val_train_splits = []

for train_index, test_index in train_test_folds:
    train_data = data[train_index]
    train_labels = labels[train_index]

    label_indices = labelled_split(train_data, train_labels, num_labelled, True)
    label_indices_list.append(label_indices)

    label_labels = train_labels[label_indices]
    lab, count = np.unique(label_labels.numpy(), return_counts=True)
    print(dict(zip(lab, count)))
    train_lab, train_count = np.unique(train_labels.numpy(), return_counts=True)
    print(dict(zip(train_lab, train_count)))

    test_val_data = data[train_index]
    test_val_labels = labels[train_index]

    val_train_splits.append(next(stratified_k_fold(test_val_data, test_val_labels, 2)))

folds_and_labels = [train_test_folds, label_indices_list, val_train_splits]

str_drop = 'drop_samples' if drop_samples else 'no_drop'
filename = './data/tcga/{}_labelled_{}_folds_{}.p'.format(num_labelled, num_folds, str_drop)
print(filename)
pickle.dump(folds_and_labels, open(filename, 'wb'))
