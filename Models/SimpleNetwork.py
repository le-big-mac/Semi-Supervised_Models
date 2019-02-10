import torch
import csv
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from utils import Datasets, Arguments, KFoldSplits, LoadData


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation):
        super(Classifier, self).__init__()

        dimensions = [input_size] + hidden_dimensions

        self.fc_layers = [nn.Sequential(
            nn.Linear(dimensions[i], dimensions[i+1]),
            activation,
        ) for i in list(range(len(dimensions)-1))]

        self.classification_layer = nn.Linear(dimensions[-1], num_classes)

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)

        return self.classification_layer(x)

class SimpleNetwork:
    def __init__(self, input_size, hidden_dimensions, num_classes, activation, device):
        self.Classifier = Classifier(input_size, hidden_dimensions, num_classes, activation).to(device)
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)
        self.Classifier_criterion = nn.CrossEntropyLoss(reduction='sum')
        self.device = device

    def train_classifier_one_epoch(self, dataloader):
        self.Classifier.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data.to(self.device)
            labels.to(self.device)

            self.Classifier_optim.zero_grad()

            preds = self.Classifier(data)

            loss = self.Classifier_criterion(preds, data)

            train_loss += loss.item()

            loss.backward()
            self.Classifier_optim.step()

        return train_loss/len(dataloader.dataset)

    def supervised_validation(self, dataloader):
        self.Classifier.eval()
        validation_loss = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data.to(self.device)
                labels.to(self.device)

                predictions = self.Classifier(data)

                loss = self.Classifier_criterion(predictions, labels)

                validation_loss += loss.item()

        return validation_loss/len(dataloader.dataset)

    def supervised_test(self, dataloader):
        self.Classifier.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                outputs = self.Classifier(data)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        return correct/len(dataloader.dataset)

    def full_train(self, train_dataset, validation_dataset):
        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        # simple early stopping employed (can change later)
        validation_result = float("inf")
        for epoch in range(50):

            self.train_classifier_one_epoch(supervised_dataloader)
            val = self.supervised_validation(validation_dataloader)

            if val > validation_result:
                break

            validation_result = val

    def full_test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return self.supervised_test(test_dataloader)


if __name__ == '__main__':

    args = Arguments.parse_args()

    _, supervised_data, supervised_labels = LoadData.load_data(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    simple_network = SimpleNetwork(500, [200], 10, nn.ReLU(), device)

    test_results = []
    for test_idx, validation_idx, train_idx in KFoldSplits.k_fold_splits_with_validation(len(unsupervised_data), 10):
        train_dataset = Datasets.SupervisedDataset([supervised_data[i] for i in train_idx],
                                                   [supervised_labels[i] for i in train_idx])
        validation_dataset = Datasets.SupervisedDataset([supervised_data[i] for i in validation_idx],
                                                        [supervised_labels[i] for i in validation_idx])
        test_dataset = Datasets.SupervisedDataset([supervised_data[i] for i in test_idx],
                                                  [supervised_labels[i] for i in test_idx])

        simple_network.full_train(train_dataset, validation_dataset)

        correct_percentage = simple_network.full_test(test_dataset)

        test_results.append(correct_percentage)

    if not os.path.exists('../results'):
        os.mkdir('../results')
        os.mkdir('../results/simple_network')
    elif not os.path.exists('../results/simple_network'):
        os.mkdir('../results/simple_network')

    accuracy_file = open('../results/simple_network/accuracy.csv', 'w')
    accuracy_writer = csv.writer(accuracy_file)

    accuracy_writer.write_row(test_results)
