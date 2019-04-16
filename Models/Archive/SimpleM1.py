import torch
from torch import nn
from Models.BuildingBlocks import Autoencoder, Classifier
from Models.Model import Model
from utils.trainingutils import EarlyStopping, unsupervised_validation_loss

# --------------------------------------------------------------------------------------
# Kingma M1 model using simple autoencoder for dimensionality reduction (for comparison)
# --------------------------------------------------------------------------------------


class SimpleM1(Model):
    def __init__(self, input_size, hidden_dimensions_encoder, latent_size, hidden_dimensions_classifier,
                 num_classes, output_activation, dataset_name, device):
        super(SimpleM1, self).__init__(dataset_name, device)

        self.Autoencoder = Autoencoder(input_size, hidden_dimensions_encoder, latent_size, lambda x: x,
                                       output_activation).to(device)
        self.Autoencoder_criterion = nn.BCELoss()
        self.Autoencoder_optim = torch.optim.Adam(self.Autoencoder.parameters(), lr=1e-3)
        self.Encoder = self.Autoencoder.encoder

        self.Classifier = Classifier(latent_size, hidden_dimensions_classifier, num_classes).to(device)
        self.Classifier_criterion = nn.CrossEntropyLoss()
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)

        self.model_name = 'simple_m1'

    def train_autoencoder(self, max_epochs, train_dataloader, validation_dataloader):
        early_stopping = EarlyStopping('{}/{}_autoencoder.pt'.format(self.model_name, self.dataset_name), patience=10)

        for epoch in range(max_epochs):
            if early_stopping.early_stop:
                break

            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_dataloader):
                self.Autoencoder.train()

                data = data.to(self.device)

                self.Autoencoder_optim.zero_grad()

                recons = self.Autoencoder(data)

                loss = self.Autoencoder_criterion(recons, data)

                train_loss += loss.item()

                loss.backward()
                self.Autoencoder_optim.step()

            validation_loss = unsupervised_validation_loss(self.Autoencoder, validation_dataloader,
                                                           self.Autoencoder_criterion, self.device)

            early_stopping(validation_loss, self.Autoencoder)

            print('Unsupervised Epoch: {} Loss: {} Validation loss: {}'.format(epoch, train_loss, validation_loss))

        if early_stopping.early_stop:
            early_stopping.load_checkpoint(self.Autoencoder)

    def train_classifier(self, max_epochs, train_dataloader, validation_dataloader, comparison):
        epochs = []
        train_losses = []
        validation_accs = []

        early_stopping = EarlyStopping('{}/{}_classifier.pt'.format(self.model_name, self.dataset_name))

        for epoch in range(max_epochs):
            if early_stopping.early_stop:
                break

            for batch_idx, (data, labels) in enumerate(train_dataloader):
                self.Classifier.train()

                data = data.float().to(self.device)
                labels = labels.to(self.device)

                self.Classifier_optim.zero_grad()

                with torch.no_grad():
                    z = self.Encoder(data)

                pred = self.Classifier(z)

                loss = self.Classifier_criterion(pred, labels)

                loss.backward()
                self.Classifier_optim.step()

                if comparison:
                    epochs.append(epoch)
                    train_losses.append(loss.item())
                    validation_accs.append(self.accuracy(validation_dataloader))

            val = self.accuracy(validation_dataloader)

            print('Supervised Epoch: {} Validation acc: {}'.format(epoch, val))

            early_stopping(1 - val, self.Classifier)

        if early_stopping.early_stop:
            early_stopping.load_checkpoint(self.Classifier)

        return epochs, train_losses, validation_accs

    def accuracy(self, dataloader):
        self.Encoder.eval()
        self.Classifier.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.float().to(self.device)
                labels = labels.to(self.device)

                z = self.Encoder(data)
                outputs = self.Classifier(z)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        return correct / len(dataloader.dataset)

    def train_model(self, max_epochs, dataloaders, comparison):
        unsupervised_dataloader, supervised_dataloader, validation_dataloader = dataloaders

        self.train_autoencoder(max_epochs, unsupervised_dataloader, validation_dataloader)

        epochs, losses, validation_accs = self.train_classifier(max_epochs, supervised_dataloader,
                                                                validation_dataloader, comparison)

        return epochs, losses, validation_accs

    def test_model(self, test_dataloader):
        return self.accuracy(test_dataloader)

    def classify(self, data):
        self.Encoder.eval()
        self.Classifier.eval()

        return self.forward(data)

    def forward(self, data):
        z = self.Encoder(data)

        return self.Classifier(z)
