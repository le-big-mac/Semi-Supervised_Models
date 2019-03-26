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
                 num_classes, latent_activation, output_activation, dataset_name, device):
        super(SimpleM1, self).__init__(dataset_name, device)

        self.Autoencoder = Autoencoder(input_size, hidden_dimensions_encoder, latent_size, latent_activation,
                                       output_activation).to(device)
        self.Autoencoder_criterion = nn.BCELoss()
        self.Autoencoder_optim = torch.optim.Adam(self.Autoencoder.parameters(), lr=1e-3)
        self.Encoder = self.Autoencoder.encoder

        self.Classifier = Classifier(latent_size, hidden_dimensions_classifier, num_classes).to(device)
        self.Classifier_criterion = nn.CrossEntropyLoss()
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)

        self.model_name = 'simple_m1'

    def train_autoencoder(self, train_dataloader, validation_dataloader):
        epochs = []
        train_losses = []
        validation_losses = []

        early_stopping = EarlyStopping('{}/{}_autoencoder.pt'.format(self.model_name, self.dataset_name), patience=5)

        epoch = 0
        while not early_stopping.early_stop:
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

            epochs.append(epoch)
            train_losses.append(train_loss/len(train_dataloader))
            validation_losses.append(validation_loss)

            print('Unsupervised Epoch: {} Loss: {} Validation loss: {}'.format(epoch, train_loss, validation_loss))

            epoch += 1

        early_stopping.load_checkpoint(self.Autoencoder)

        return epochs, train_losses, validation_losses

    def train_classifier(self, train_dataloader, validation_dataloader):
        epochs = []
        train_losses = []
        validation_accs = []

        early_stopping = EarlyStopping('{}/{}_classifier.pt'.format(self.model_name, self.dataset_name))

        epoch = 0
        while not early_stopping.early_stop:
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

                validation_acc = self.accuracy(validation_dataloader)

                epochs.append(epoch)
                train_losses.append(loss.item())
                validation_accs.append(validation_acc)

                print('Supervised Epoch: {} Loss: {} Validation acc: {}'.format(epoch, loss.item(), validation_acc))

            early_stopping(1 - sum(validation_accs)/len(validation_accs), self.Classifier)

            epoch += 1

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

    def train(self, supervised_dataloader, unsupervised_dataloader, validation_dataloader=None):
        autoencoder_epochs, autoencoder_train_losses, autoencoder_validation_losses = \
            self.train_autoencoder(unsupervised_dataloader, validation_dataloader)

        classifier_epochs, classifier_losses, classifier_accs = \
            self.train_classifier(supervised_dataloader, validation_dataloader)

        return classifier_epochs, classifier_losses, classifier_accs

    def test(self, test_dataloader):
        return self.accuracy(test_dataloader)
