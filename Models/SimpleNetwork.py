import torch
import csv
import os
from torch import nn
from torch.utils.data import DataLoader
from utils import Accuracy, Arguments, KFoldSplits, Datasets, LoadData


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation):
        super(Classifier, self).__init__()

        dimensions = [input_size] + hidden_dimensions

        layers = [
            nn.Sequential(
                nn.Linear(dimensions[i-1], dimensions[i]),
                activation,
            )
            for i in range(1, len(dimensions))
        ]

        self.fc_layers = nn.ModuleList(layers)

        self.classification_layer = nn.Linear(dimensions[-1], num_classes)

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)

        return self.classification_layer(x)

class SimpleNetwork:
    def __init__(self, input_size, hidden_dimensions, num_classes, activation, device):
        self.Classifier = Classifier(input_size, hidden_dimensions, num_classes, activation).to(device)
        self.optimizer = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.state_path = 'state/simple_mlp.pt'
        self.device = device
        os.remove(self.state_path)
        torch.save(self.Classifier.state_dict(), self.state_path)

    def train_classifier_one_epoch(self, epoch, dataloader):
        self.Classifier.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            preds = self.Classifier(data)

            loss = self.criterion(preds, labels)

            train_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        print('Epoch: {} Loss: {}'.format(epoch, train_loss/len(dataloader.dataset)))

        return train_loss/len(dataloader.dataset)

    def reset_model(self):
        self.Classifier.load_state_dict(torch.load(self.state_path))

        self.optimizer = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)

    def full_train(self, train_dataset, validation_dataset):
        self.reset_model()

        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(50):

            self.train_classifier_one_epoch(epoch, supervised_dataloader)
            print('Epoch: {} Validation Acc: {}'.format(epoch, Accuracy.accuracy(self.Classifier, validation_dataloader,
                                                                                 self.device)))

    def full_test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return Accuracy.accuracy(self.Classifier, test_dataloader, self.device)


def MNIST_train(device):

    _, supervised_dataset, validation_dataset, test_dataset = LoadData.load_MNIST_data(100, 10000, 10000, 0)

    network = SimpleNetwork(784, [1000, 500, 250, 250, 250], 10, nn.ReLU(), device)

    network.full_train(supervised_dataset, validation_dataset)

    return network.full_test(test_dataset)


def file_train(device):

    args = Arguments.parse_args()

    _, supervised_data, supervised_labels = LoadData.load_data_from_file(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    simple_network = SimpleNetwork(500, [200], 10, nn.ReLU(), device)

    test_results = []
    for test_idx, train_idx in KFoldSplits.k_fold_splits(len(supervised_data), 10):
        train_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
                                                                 [supervised_labels[i] for i in train_idx])
        test_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
                                                                [supervised_labels[i] for i in test_idx])

        simple_network.full_train(train_dataset)

        correct_percentage = simple_network.full_test(test_dataset)

        test_results.append(correct_percentage)

    if not os.path.exists('../results'):
        os.mkdir('../results')
        os.mkdir('../results/simple_network')
    elif not os.path.exists('../results/simple_network'):
        os.mkdir('../results/simple_network')

    accuracy_file = open('../results/simple_network/accuracy.csv', 'w')
    accuracy_writer = csv.writer(accuracy_file)

    accuracy_writer.writerow(test_results)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MNIST_train(device)
