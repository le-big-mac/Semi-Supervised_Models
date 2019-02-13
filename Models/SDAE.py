import torch
import os
import csv
from torch import nn
from torch.utils.data import DataLoader
from utils import Datasets, LoadData, Arguments, KFoldSplits


class Encoder(nn.Module):
    def __init__(self, input_size, layer_size, activation):
        super(Encoder, self).__init__()

        self.layer = nn.Linear(input_size, layer_size)
        self.activation = activation

    def forward(self, x):
        h = self.layer(x)
        if self.activation:
            h = self.activation(h)

        return h


class DAE(nn.Module):
    def __init__(self, encoder):
        super(DAE, self).__init__()

        self.encoder = encoder
        self.decoder = nn.Linear(encoder.layer.out_features, encoder.layer.in_feature)

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)


class PretrainedSDAE(nn.Module):
    def __init__(self, layers, num_classes, activation):
        super(PretrainedSDAE, self).__init__()

        self.activation = activation
        self.pretrained_hidden_fc_layers = layers
        self.classification_layer = nn.Linear(layers[-1].out_features, num_classes)

    def forward(self, x):
        cur = x

        for layer in self.pretrained_hidden_fc_layers:
            cur = self.activation(layer(cur))

        return self.classification_layer(cur)


class SDAE:
    def __init__(self, hidden_dimensions, input_size, num_classes, activation, device):
        self.hidden_dimensions = hidden_dimensions
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_layers = []
        self.SDAE = self.setup_model(hidden_dimensions, input_size, num_classes, activation)
        self.device = device

    def setup_model(self, hidden_dimensions, input_size, num_classes, activation):
        self.hidden_layers.append(Encoder(input_size=input_size, layer_size=hidden_dimensions[0],
                                          activation=activation))

        for i in range(1, len(hidden_dimensions)):
            self.hidden_layers.append(Encoder(input_size=hidden_dimensions[i-1], layer_size=hidden_dimensions[i],
                                              activation=activation))

        return PretrainedSDAE(self.hidden_layers, num_classes, activation).to(self.device)

    def pretrain_hidden_layers(self, dataloader):

        for i in range(len(self.hidden_layers)):
            decoder = DAE(self.hidden_layers[0]).to(self.device)
            criterion = nn.MSELoss(reduction="sum")
            optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

            previous_layers = None
            if i > 0:
                previous_layers = self.hidden_layers[0:i-1]

            for epoch in range(50):
                self.train_DAE_one_epoch(previous_layers, decoder, dataloader, criterion, optimizer)

    def train_DAE_one_epoch(self, previous_layers, dae, dataloader, criterion, optimizer):
        dae.train()
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data.to(self.device)

            if previous_layers:
                with torch.no_grad():
                    for layer in previous_layers:
                        data = layer(data)

            noisy_data = data.add(0.2*torch.randn(data.size()).to(self.device))

            optimizer.zero_grad()

            predictions = dae(noisy_data)

            loss = criterion(predictions, data)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        return train_loss/len(dataloader.dataset)

    def supervised_train_one_epoch(self, dataloader, criterion, optimizer):
        model = self.SDAE

        model.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data.to(self.device)
            labels.to(self.device)

            optimizer.zero_grad()

            predictions = model(data)

            loss = criterion(predictions, data)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        return train_loss/len(dataloader.dataset)

    def supervised_validation(self, dataloader, criterion):
        model = self.SDAE

        model.eval()
        validation_loss = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data.to(self.device)
                labels.to(self.device)

                predictions = model(data)

                loss = criterion(predictions, labels)

                validation_loss += loss.item()

        return validation_loss/len(dataloader.dataset)

    def supervised_test(self, dataloader):
        model = self.SDAE

        model.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        return correct/len(dataloader.dataset)

    def full_train(self, unsupervised_dataset, train_dataset, validation_dataset):

        combined_dataset = Datasets.UnsupervisedDataset(unsupervised_dataset.raw_data + train_dataset.raw_input)

        pretraining_dataloader = DataLoader(dataset=combined_dataset, batch_size=200, shuffle=True)

        self.pretrain_hidden_layers(pretraining_dataloader)

        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.Adam(self.SDAE.parameters(), lr=1e-3)

        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        # simple early stopping employed (can change later)

        validation_result = float("inf")
        for epoch in range(50):

            self.supervised_train_one_epoch(supervised_dataloader, criterion, optimizer)
            val = self.supervised_validation(validation_dataloader, criterion)

            if val > validation_result:
                break

            validation_result = val

    def full_test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return self.supervised_test(test_dataloader)


if __name__ == '__main__':

    args = Arguments.parse_args()

    unsupervised_data, supervised_data, supervised_labels = LoadData.load_data(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sdae = SDAE([200], 500, 10, nn.ReLU(), device)

    unsupervised_dataset = Datasets.UnsupervisedDataset(unsupervised_data)

    test_results = []

    for test_idx, validation_idx, train_idx in KFoldSplits.k_fold_splits_with_validation(len(unsupervised_data), 10):
        train_dataset = Datasets.SupervisedDataset([supervised_data[i] for i in train_idx],
                                                   [supervised_labels[i] for i in train_idx])
        validation_dataset = Datasets.SupervisedDataset([supervised_data[i] for i in validation_idx],
                                                        [supervised_labels[i] for i in validation_idx])
        test_dataset = Datasets.SupervisedDataset([supervised_data[i] for i in test_idx],
                                                  [supervised_labels[i] for i in test_idx])

        sdae.full_train(unsupervised_data, train_dataset, validation_dataset)

        correct_percentage = sdae.full_test(test_dataset)

        test_results.append(correct_percentage)

    if not os.path.exists('../results'):
        os.mkdir('../results')
        os.mkdir('../results/sdae')
    elif not os.path.exists('../results/sdae'):
        os.mkdir('../results/sdae')

    accuracy_file = open('../results/sdae/accuracy.csv', 'w')
    accuracy_writer = csv.writer(accuracy_file)

    accuracy_writer.write_row(test_results)

