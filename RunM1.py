import torch
from torch import nn
from Models.M1 import M1
from utils import Arguments, LoadData, Datasets, KFoldSplits, SaveResults


if __name__ == '__main__':

    args = Arguments.parse_args()

    unsupervised_data, supervised_data, supervised_labels = LoadData.load_data_from_file(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    m1 = M1([200], 50, 500, 10, nn.ReLU(), device)

    unsupervised_dataset = Datasets.UnsupervisedDataset(unsupervised_data)

    test_results = []
    for test_idx, train_idx in KFoldSplits.k_fold_splits(len(supervised_data), 10):
        train_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
                                                                 [supervised_labels[i] for i in train_idx])
        test_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
                                                                [supervised_labels[i] for i in test_idx])

        m1.full_train(unsupervised_dataset, train_dataset)

        correct_percentage = m1.full_test(test_dataset)

        test_results.append(correct_percentage)

    SaveResults.save_results([test_results], 'm1')
