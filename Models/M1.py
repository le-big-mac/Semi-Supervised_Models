import torch
from torch import nn
from torch import functional as F
from torch.utils.data import DataLoader
from utils import Datasets, LoadData


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dimensions, activation):
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

        self.mu = nn.Linear(hidden_dimensions[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dimensions[-1], latent_dim)

    def encode(self, x):
        for layer in self.fc_layers:
            x = layer(x)

        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn()
        z = eps.mul(std).add(mu)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar), mu, logvar


class VAE(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dimensions, activation):
        super(VAE, self).__init__()

        self.encoder = Encoder(input_size, latent_dim, hidden_dimensions, activation)
        self.fc_layers = []
        self.fc_layers.append(
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dimensions[-1]),
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

        self.out = nn.Sequential(
            nn.Linear(hidden_dimensions[0], input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z, mu, logvar = self.encoder(x)

        h = z
        for layer in self.fc_layers:
            h = layer(h)

        out = self.out(h)

        return out, mu, logvar


class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()

        # could change this to more layers
        self.supervised_layer = nn.Linear(latent_dim, num_classes)

    def forward(self, z):
        return self.supervised_layer(z)


class M1:
    def __init__(self, hidden_dimensions, latent_size, input_size, num_classes, activation, device):
        self.VAE = VAE(input_size, latent_size, hidden_dimensions, activation).to(device)
        self.Encoder = self.VAE.encoder
        self.Classifier = Classifier(latent_size, num_classes).to(device)
        self.Classifier_criterion = nn.CrossEntropyLoss(reduction='sum')
        self.VAE_optim = torch.optim.Adam(self.VAE.parameters(), lr=1e-3)
        self.Classifier_optim = torch.optim.Adam(self.Classifier.supervised_layer.parameters(), lr=1e-3)
        self.device = device

    def VAE_criterion(self, pred_x, x, mu, logvar):
        # difference between two normal distributions (N(0, 1) and parameterized)
        # TODO: is this still a tensor?
        KLD = 0.5*torch.sum(logvar.exp() + mu.pow(2) - logvar - 1)

        # reconstruction error (use BCE because we normalize input data to [0, 1] and sigmoid output)
        BCE = F.binary_cross_entropy(pred_x, x, reduction="sum")

        return KLD + BCE

    def train_VAE_one_epoch(self, dataloader):
        self.VAE.train()
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data.to(self.device)

            self.VAE_optim.zero_grad()

            recons, mu, logvar = self.VAE(data)

            loss = self.VAE_criterion(recons, data, mu, logvar)

            train_loss += loss.item()

            loss.backward()
            self.VAE_optim.step()

        return train_loss/len(dataloader.dataset)

    def pretrain_VAE(self, dataloader):

        for epoch in range(50):
            self.train_VAE_one_epoch(dataloader)

    def train_classifier_one_epoch(self, dataloader):
        self.Classifier.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data.to(self.device)
            labels.to(self.device)

            with torch.no_grad():
                z, _, _ = self.Encoder(data)

            pred = self.Classifier(z)

            loss = self.Classifier_criterion(pred, labels)

            train_loss += loss.item()

            loss.backward()
            self.Classifier_optim.step()

        return train_loss/len(dataloader.dataset)

    def supervised_validation(self, dataloader):
        self.Encoder.eval()
        self.Classifier.eval()
        validation_loss = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data.to(self.device)
                labels.to(self.device)

                z, _, _ = self.Encoder(data)
                predictions = self.Classifier(z)

                loss = self.Classifier_criterion(predictions, labels)

                validation_loss += loss.item()

        return validation_loss/len(dataloader.dataset)

    def supervised_test(self, dataloader):
        self.Encoder.eval()
        self.Classifier.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                z, _, _ = self.Encoder(data)
                outputs = self.Classifier(z)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        return correct/len(dataloader.dataset)

    def full_train(self, unsupervised_dataset, train_dataset, validation_dataset):

        combined_dataset = Datasets.UnsupervisedDataset(unsupervised_dataset.raw_data + train_dataset.raw_input)

        pretraining_dataloader = DataLoader(dataset=combined_dataset, batch_size=200, shuffle=True)

        self.pretrain_VAE(pretraining_dataloader)

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

