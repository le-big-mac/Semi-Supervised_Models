import torch
import os
import csv
from torch import nn
from torch.utils.data import DataLoader
from Models.utils import Accuracy, Arguments, KFoldSplits, Datasets, LoadData


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
        self.decoder = nn.Linear(encoder.layer.out_features, encoder.layer.in_features)

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)


class PretrainedSDAE(nn.Module):
    def __init__(self, layers, num_classes):
        super(PretrainedSDAE, self).__init__()

        self.pretrained_hidden_fc_layers = layers
        self.classification_layer = nn.Linear(layers[-1].layer.out_features, num_classes)

    def forward(self, x):
        for layer in self.pretrained_hidden_fc_layers:
            x = layer(x)

        return self.classification_layer(x)


class SDAE:
    def __init__(self, input_size, hidden_dimensions, num_classes, activation, device):
        self.hidden_dimensions = hidden_dimensions
        self.hidden_layers = nn.ModuleList()
        self.device = device
        self.PretrainedSDAE = self.setup_model(hidden_dimensions, input_size, num_classes, activation)
        self.optimizer = torch.optim.Adam(self.PretrainedSDAE.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.state_path = 'state/sdae.pt'
        os.remove(self.state_path)
        torch.save(self.PretrainedSDAE.state_dict(), self.state_path)

    def setup_model(self, hidden_dimensions, input_size, num_classes, activation):
        dims = [input_size] + hidden_dimensions

        layers = [Encoder(input_size=dims[i-1], layer_size=dims[i], activation=activation)
                  for i in range(1, len(dims))]

        self.hidden_layers = nn.ModuleList(layers)
        return PretrainedSDAE(self.hidden_layers, num_classes).to(self.device)

    def pretrain_hidden_layers(self, dataloader):

        for i in range(len(self.hidden_layers)):
            decoder = DAE(self.hidden_layers[i]).to(self.device)
            criterion = nn.MSELoss(reduction="sum")
            optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

            previous_layers = self.hidden_layers[0:i]

            for epoch in range(20):
                layer_loss = self.train_DAE_one_epoch(previous_layers, decoder, dataloader, criterion, optimizer)
                if epoch % 10 == 0:
                    print('Epoch: {} Layer loss: {}'.format(epoch, layer_loss))

    def train_DAE_one_epoch(self, previous_layers, dae, dataloader, criterion, optimizer):
        dae.train()
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data.to(self.device)

            with torch.no_grad():
                for layer in previous_layers:
                    data = layer(data)

            noisy_data = data.add(0.2*torch.randn_like(data).to(self.device))

            optimizer.zero_grad()

            predictions = dae(noisy_data)

            loss = criterion(predictions, data)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        return train_loss/len(dataloader.dataset)

    def supervised_train_one_epoch(self, epoch, dataloader):
        self.PretrainedSDAE.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data.to(self.device)
            labels.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.PretrainedSDAE(data)

            loss = self.criterion(predictions, labels)

            train_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        if epoch % 10 == 0:
            print("Epoch: {} Supervised Loss: {}".format(epoch, train_loss/len(dataloader.dataset)))

        return train_loss/len(dataloader.dataset)

    def reset_model(self):
        self.PretrainedSDAE.load_state_dict(torch.load(self.state_path))

        self.optimizer = torch.optim.Adam(self.PretrainedSDAE.parameters(), lr=1e-3)

    def full_train(self, combined_dataset, train_dataset, validation_dataset):
        self.reset_model()
        pretraining_dataloader = DataLoader(dataset=combined_dataset, batch_size=1000, shuffle=True)

        self.pretrain_hidden_layers(pretraining_dataloader)

        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)

        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(50):

            self.supervised_train_one_epoch(epoch, supervised_dataloader)
            print('Epoch: {} Validation Acc: {}'.format(epoch, Accuracy.accuracy(self.PretrainedSDAE,
                                                                                 validation_dataloader)))

    def full_test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return Accuracy.accuracy(self.PretrainedSDAE, test_dataloader)


def MNIST_train(device):

    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
        LoadData.load_MNIST_data(100, 10000, 10000, 49000)

    combined_dataset = Datasets.MNISTUnsupervised(torch.cat((unsupervised_dataset.data, supervised_dataset.data), 0))

    sdae = SDAE(784, [1000, 500, 250, 250, 250], 10, nn.ReLU(), device)

    print(sdae.PretrainedSDAE)

    sdae.full_train(combined_dataset, supervised_dataset, validation_dataset)

    return sdae.full_test(test_dataset)


def file_train(device):

    args = Arguments.parse_args()

    unsupervised_data, supervised_data, supervised_labels = LoadData.load_data_from_file(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sdae = SDAE(500, [200], 10, nn.ReLU(), device)

    test_results = []
    for test_idx, train_idx in KFoldSplits.k_fold_splits(len(supervised_data), 10):
        train_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
                                                                 [supervised_labels[i] for i in train_idx])
        test_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
                                                                [supervised_labels[i] for i in test_idx])

        sdae.full_train(train_dataset)

        correct_percentage = sdae.full_test(test_dataset)

        test_results.append(correct_percentage)

    if not os.path.exists('../results'):
        os.mkdir('../results')
        os.mkdir('../results/sdae')
    elif not os.path.exists('../results/sdae'):
        os.mkdir('../results/sdae')

    accuracy_file = open('../results/sdae/accuracy.csv', 'w')
    accuracy_writer = csv.writer(accuracy_file)

    accuracy_writer.writerow(test_results)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MNIST_train(device)

