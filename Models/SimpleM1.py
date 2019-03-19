import torch
from torch import nn
from torch.utils.data import DataLoader
from Models.BuildingBlocks import Autoencoder
from Models.Model import Model

# --------------------------------------------------------------------------------------
# Kingma M1 model using simple autoencoder for dimensionality reduction (for comparison)
# --------------------------------------------------------------------------------------


class Classifier(nn.Module):
    def __init__(self, latent_dim, hidden_dimensions_classifier, num_classes):
        super(Classifier, self).__init__()

        dims = [latent_dim] + hidden_dimensions_classifier

        layers = [nn.Sequential(
            nn.Linear(dims[i-1], dims[i]),
            nn.ReLU(),
        ) for i in range(1, len(dims))]

        self.layers = nn.ModuleList(layers)
        self.classification = nn.Linear(dims[-1], num_classes)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)

        return self.classification(z)


class SimpleM1(Model):
    def __init__(self, input_size, hidden_dimensions_encoder, latent_size, hidden_dimensions_classifier,
                 num_classes, activation, device):
        super(SimpleM1, self).__init__(device)

        self.Autoencoder = Autoencoder(input_size, hidden_dimensions_encoder, latent_size, activation).to(device)
        self.Autoencoder_criterion = nn.BCELoss(reduction='sum')
        self.Autoencoder_optim = torch.optim.Adam(self.Autoencoder.parameters(), lr=1e-3)
        self.Encoder = self.Autoencoder.encoder

        self.Classifier = Classifier(latent_size, hidden_dimensions_classifier, num_classes).to(device)
        self.Classifier_criterion = nn.CrossEntropyLoss(reduction='sum')
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)

    def train_autoencoder_one_epoch(self, epoch, dataloader):
        self.Autoencoder.train()
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data = data.to(self.device)

            self.VAE_optim.zero_grad()

            recons = self.Autoencoder(data)

            loss = self.Autoencoder_criterion(recons, data)

            train_loss += loss.item()

            loss.backward()
            self.Autoencoder_optim.step()

        print('Epoch: {} VAE Loss: {}'.format(epoch, train_loss/len(dataloader.dataset)))

        return train_loss/len(dataloader.dataset)

    def train_classifier_one_epoch(self, epoch, dataloader):
        self.Classifier.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.Classifier_optim.zero_grad()

            with torch.no_grad():
                z, _, _ = self.Encoder(data)

            pred = self.Classifier(z)

            loss = self.Classifier_criterion(pred, labels)

            train_loss += loss.item()

            loss.backward()
            self.Classifier_optim.step()

        print('Epoch {} Loss {}:'.format(epoch, train_loss/len(dataloader.dataset)))

        return train_loss/len(dataloader.dataset)

    def supervised_test(self, dataloader):
        self.Encoder.eval()
        self.Classifier.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                z = self.Encoder(data)
                outputs = self.Classifier(z)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        return correct / len(dataloader.dataset)

    def train(self, combined_dataset, train_dataset, validation_dataset=None):
        pretraining_dataloader = DataLoader(dataset=combined_dataset, batch_size=1000, shuffle=True)

        for epoch in range(50):
            self.train_autoencoder_one_epoch(epoch, pretraining_dataloader)

        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(50):
            self.train_classifier_one_epoch(epoch, supervised_dataloader)
            print('Epoch: {} Validation Acc: {}'.format(epoch, self.supervised_test(validation_dataloader)))

    def test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return self.supervised_test(test_dataloader)
