import torch
from Models.Ladder import LadderNetwork
from utils import arguments, datautils


def MNIST_train(device):

    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
        datautils.load_MNIST_data(100, num_unlabelled=49900, validation=True, test=True)

    results = []
    for i in range(5):
        ladder = LadderNetwork(784, [1000, 500, 250, 250, 250], 10, [0.1, 0.1, 0.1, 0.1, 0.1, 10, 1000], device, 0.3)

        print(ladder.ladder)

        ladder.train(unsupervised_dataset, supervised_dataset, validation_dataset)

        results.append(ladder.test(test_dataset))

    datautils.save_results(results, 'ladder', 'MNIST_accuracy')


def file_train(device):

    args = arguments.parse_args()

    unsupervised_data, supervised_data, supervised_labels = datautils.load_data_from_file(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ladder = LadderNetwork(784, [1000, 500, 250, 250, 250], 10, ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax'],
                               0.2, [1000, 10, 0.1, 0.1, 0.1, 0.1, 0.1], device)

    test_results = []
    for test_idx, train_idx in datautils.k_fold_splits(len(supervised_data), 10):
        train_dataset = datautils.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
                                                                  [supervised_labels[i] for i in train_idx])
        test_dataset = datautils.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
                                                                 [supervised_labels[i] for i in test_idx])

        ladder.train(train_dataset)

        correct_percentage = ladder.test(test_dataset)

        test_results.append(correct_percentage)

    datautils.save_results([test_results], 'ladder', 'accuracy')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MNIST_train(device)
