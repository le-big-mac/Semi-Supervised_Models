import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import Datasets, LoadData


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation):
        super(Encoder, self).__init__()

        self.fc_layers = []
        self.fc_layers.append(
            nn.Sequential(
                nn.Linear(input_size, hidden_dimensions[0]),
                activation,
            )
        )

        for i in range(1, len(hidden_dimensions)):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dimensions[i-1], hidden_dimensions[i]),
                    activation,
                )
            )

        self.classification_layer = nn.Linear(hidden_dimensions[-1], num_classes)

    def forward(self, x):
        return self.encoder(x)


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(input_size, hidden_dimensions, num_classes, activation)
        self.fc_layers = []
        self.fc_layers.append(
            nn.Sequential(
                nn.Linear(num_classes, hidden_dimensions[-1]),
                activation,
            )
        )

        for i in range(len(hidden_dimensions), 1, -1):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dimensions[i], hidden_dimensions[i-1]),
                    activation,
                )
            )

        # sigmoid maybe not necessary (constraining input to 0-1)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dimensions[0], input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


class DeepMetabolism:
    def __init__(self, input_size, hidden_dimensions, num_classes, activation, device):
        self.Autoencoder = Autoencoder(input_size, hidden_dimensions, num_classes, activation).to(device)
        self.Autoencoder_optim = torch.optim.Adam(self.Autoencoder.parameters(), lr=1e-3)
        self.Autoencoder_criterion = nn.MSELoss(reduction='sum')
        self.Classifier = self.Autoencoder.encoder
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)
        self.Classifier_criterion = nn.CrossEntropyLoss(reduction='sum')
        self.device = device

    def unsupervised_train_one_epoch(self, dataloader):
        self.Autoencoder.train()
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data.to(self.device)

            self.Autoencoder_optim.zero_grad()

            recons = self.Autoencoder(data)

            loss = self.Autoencoder_criterion(recons, data)

            train_loss += loss.item()

            loss.backward()
            self.Autoencoder_optim.step()

        return train_loss/len(dataloader.dataset)

    def pretrain_classifier(self, dataloader):
        for epoch in range(50):
            self.unsupervised_train_one_epoch(dataloader)

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

    def full_train(self, unsupervised_dataset, train_dataset, validation_dataset):

        combined_dataset = Datasets.UnsupervisedDataset(unsupervised_dataset.raw_data + train_dataset.raw_input)

        pretraining_dataloader = DataLoader(dataset=combined_dataset, batch_size=200, shuffle=True)

        self.pretrain_classifier(pretraining_dataloader)

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

