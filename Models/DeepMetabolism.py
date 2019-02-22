import torch
import os
import csv
from torch import nn
from torch.utils.data import DataLoader
from utils import Accuracy, Arguments, KFoldSplits, Datasets, LoadData


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation):
        super(Encoder, self).__init__()

        dims = [input_size] + hidden_dimensions

        layers = [
            nn.Sequential(
                nn.Linear(dims[i-1], dims[i]),
                activation,
            )
            for i in range(1, len(dims))
        ]

        self.fc_layers = nn.ModuleList(layers)
        self.classification_layer = nn.Linear(hidden_dimensions[-1], num_classes)

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)

        return self.classification_layer(x)


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(input_size, hidden_dimensions, num_classes, activation)

        dims = hidden_dimensions + [num_classes]

        layers = [
            nn.Sequential(
                nn.Linear(dims[i], dims[i-1]),
                activation,
            )
            for i in range(len(dims)-1, 0, -1)
        ]

        self.fc_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dimensions[0], input_size)

    def forward(self, x):
        h = self.encoder(x)

        for layer in self.fc_layers:
            h = layer(h)

        return self.output_layer(h)


class DeepMetabolism:
    def __init__(self, input_size, hidden_dimensions, num_classes, activation, device):
        self.Autoencoder = Autoencoder(input_size, hidden_dimensions, num_classes, activation).to(device)
        self.Autoencoder_optim = torch.optim.Adam(self.Autoencoder.parameters(), lr=1e-3)
        self.Autoencoder_criterion = nn.MSELoss(reduction='sum')
        self.Classifier = self.Autoencoder.encoder
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)
        self.Classifier_criterion = nn.CrossEntropyLoss(reduction='sum')
        self.device = device
        self.state_path = 'state/deep_metabolism.pt'
        torch.save(self.Autoencoder.state_dict(), self.state_path)

    def unsupervised_train_one_epoch(self, dataloader):
        self.Autoencoder.train()
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data = data.to(self.device)

            self.Autoencoder_optim.zero_grad()

            recons = self.Autoencoder(data)

            loss = self.Autoencoder_criterion(recons, data)

            train_loss += loss.item()

            loss.backward()
            self.Autoencoder_optim.step()

        print('Unsupervised Loss: {}'.format(train_loss/len(dataloader.dataset)))

        return train_loss/len(dataloader.dataset)

    def pretrain_classifier(self, dataloader):
        for epoch in range(50):
            self.unsupervised_train_one_epoch(dataloader)

    def train_classifier_one_epoch(self, epoch, dataloader):
        self.Classifier.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.Classifier_optim.zero_grad()

            preds = self.Classifier(data)

            loss = self.Classifier_criterion(preds, labels)

            train_loss += loss.item()

            loss.backward()
            self.Classifier_optim.step()

        print('Epoch: {} Loss: {}'.format(epoch, train_loss/len(dataloader.dataset)))

        return train_loss/len(dataloader.dataset)

    def reset_model(self):
        self.Autoencoder.load_state_dict(torch.load(self.state_path))
        self.Classifier = self.Autoencoder.encoder

        self.Autoencoder_optim = torch.optim.Adam(self.Autoencoder.parameters(), lr=1e-3)
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)

    def full_train(self, combined_dataset, train_dataset, validation_dataset=None):
        self.reset_model()

        pretraining_dataloader = DataLoader(dataset=combined_dataset, batch_size=1000, shuffle=True)

        self.pretrain_classifier(pretraining_dataloader)

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

    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
        LoadData.load_MNIST_data(100, 10000, 10000, 49000)

    combined_dataset = Datasets.MNISTUnsupervised(torch.cat((unsupervised_dataset.data, supervised_dataset.data), 0))

    deep_metabolism = DeepMetabolism(784, [1000, 500, 250, 250, 250], 10, nn.ReLU(), device)

    print(deep_metabolism.Classifier)

    deep_metabolism.full_train(combined_dataset, supervised_dataset, validation_dataset)

    return deep_metabolism.full_test(test_dataset)


def file_train(device):

    args = Arguments.parse_args()

    unsupervised_data, supervised_data, supervised_labels = LoadData.load_data_from_file(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    deep_metabolism = DeepMetabolism(500, [200], 10, nn.ReLU(), device)

    test_results = []
    for test_idx, train_idx in KFoldSplits.k_fold_splits(len(supervised_data), 10):
        train_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
                                                                 [supervised_labels[i] for i in train_idx])
        test_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
                                                                [supervised_labels[i] for i in test_idx])

        deep_metabolism.full_train(train_dataset)

        correct_percentage = deep_metabolism.full_test(test_dataset)

        test_results.append(correct_percentage)

    if not os.path.exists('../results'):
        os.mkdir('../results')
        os.mkdir('../results/deep_metabolism')
    elif not os.path.exists('../results/deep_metabolism'):
        os.mkdir('../results/deep_metabolism')

    accuracy_file = open('../results/deep_metabolism/accuracy.csv', 'w')
    accuracy_writer = csv.writer(accuracy_file)

    accuracy_writer.writerow(test_results)


if __name__ == '__main__':

    MNIST_train('cpu')
