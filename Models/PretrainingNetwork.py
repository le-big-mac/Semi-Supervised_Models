import torch
from torch import nn
from utils.trainingutils import accuracy, unsupervised_validation_loss
from Models.BuildingBlocks import Autoencoder
from Models.Model import Model
from utils.trainingutils import EarlyStopping


class PretrainingNetwork(Model):
    def __init__(self, input_size, hidden_dimensions, num_classes, latent_activation, output_activation, dataset_name,
                 device):
        super(PretrainingNetwork, self).__init__(dataset_name, device)

        self.Autoencoder = Autoencoder(input_size, hidden_dimensions, num_classes, latent_activation,
                                       output_activation).to(device)
        self.Autoencoder_optim = torch.optim.Adam(self.Autoencoder.parameters(), lr=1e-3)
        self.Autoencoder_criterion = nn.MSELoss()

        self.Classifier = self.Autoencoder.encoder
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)
        self.Classifier_criterion = nn.CrossEntropyLoss()

        self.model_name = 'pretraining'

    def train_autoencoder(self, train_dataloader, validation_dataloader):
        epochs = []
        train_losses = []
        validation_losses = []

        early_stopping = EarlyStopping('{}/{}_autoencoder.pt'.format(self.model_name, self.dataset_name), patience=7)

        epoch = 0
        while not early_stopping.early_stop:
        # while epoch < 50:
            train_loss = 0
            validation_loss = 0
            for batch_idx, (data, _) in enumerate(train_dataloader):
                self.Autoencoder.train()

                data = data.to(self.device)

                self.Autoencoder_optim.zero_grad()

                recons = self.Autoencoder(data)

                loss = self.Autoencoder_criterion(recons, data)

                train_loss += loss.item()

                loss.backward()
                self.Autoencoder_optim.step()

                validation_loss += unsupervised_validation_loss(self.Autoencoder, validation_dataloader,
                                                                self.Autoencoder_criterion, self.device)

            early_stopping(validation_loss, self.Autoencoder)

            epochs.append(epoch)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            print('Unsupervised Epoch: {} Loss: {} Validation loss: {}'.format(epoch, train_loss, validation_loss))
            # print('Unsupervised Loss: {}'.format(train_loss))

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
        # while epoch < 50:
            for batch_idx, (data, labels) in enumerate(train_dataloader):
                self.Classifier.train()

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.Classifier_optim.zero_grad()

                preds = self.Classifier(data)

                loss = self.Classifier_criterion(preds, labels)

                loss.backward()
                self.Classifier_optim.step()

                validation_acc = accuracy(self.Classifier, validation_dataloader, self.device)

                epochs.append(epoch)
                train_losses.append(loss.item())
                validation_accs.append(validation_acc)

                print('Supervised Epoch: {} Loss: {} Validation acc: {}'.format(epoch, loss.item(), validation_acc))

            early_stopping(1 - sum(validation_accs)/len(validation_accs), self.Classifier)

            epoch += 1

        early_stopping.load_checkpoint(self.Classifier)

        return epochs, train_losses, validation_accs

    def train(self, supervised_dataloader, unsupervised_dataloader, validation_dataloader=None):
        autoencoder_epochs, autoencoder_train_losses, autoencoder_validation_losses = \
            self.train_autoencoder(unsupervised_dataloader, validation_dataloader)

        classifier_epochs, classifier_train_losses, classifier_validation_accs = \
            self.train_classifier(supervised_dataloader, validation_dataloader)

        return classifier_epochs, classifier_train_losses, classifier_validation_accs

    def test(self, test_dataloader):
        return accuracy(self.Classifier, test_dataloader, self.device)
