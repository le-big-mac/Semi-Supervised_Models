from utils.datautils import *
import pickle
import argparse


parser = argparse.ArgumentParser(description='Take arguments to construct model')
parser.add_argument('drop_samples', type=bool, help='Get indexes for data with samples with missing genes dropped')
parser.add_argument('num_labelled', type=int, help='Number of labelled examples to use')
parser.add_argument('num_folds', type=int, help='Number of folds')
args = parser.parse_args()
drop_samples = args.drop_samples
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

    label_indices_list.append(labelled_split(train_data, train_labels, num_labelled, True))

    test_val_data = data[train_index]
    test_val_labels = labels[train_index]

    val_train_splits.append(next(stratified_k_fold(test_val_data, test_val_labels, 2)))

folds_and_labels = [train_test_folds, label_indices_list, val_train_splits]

str_drop = 'drop_samples' if drop_samples else 'nodrop'
pickle.dump(folds_and_labels, open('./data/tcga/{}_labelled_{}_folds_{}.p'.format(num_labelled, num_folds,
                                                                                  drop_samples), 'wb'))
