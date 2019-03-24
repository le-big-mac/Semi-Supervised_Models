import torch
from torch import nn
from utils.trainingutils import accuracy
from Models.BuildingBlocks import Encoder, AutoencoderSDAE
from Models.Model import Model
from utils.trainingutils import EarlyStopping


class SDAEClassifier(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation):
        super(SDAEClassifier, self).__init__()

        dims = [input_size] + hidden_dimensions

        layers = [Encoder(dims[i], [], dims[i+1], activation)
                  for i in range(0, len(dims)-1)]

        self.hidden_layers = nn.ModuleList(layers)
        self.classification_layer = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        return self.classification_layer(x)


class SDAE(Model):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation, dataset_name, device):
        super(SDAE, self).__init__(dataset_name, device)

        self.SDAEClassifier = SDAEClassifier(input_size, hidden_dimensions, num_classes, activation).to(device)
        self.optimizer = torch.optim.Adam(self.SDAEClassifier.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        self.model_name = 'sdae'

    def pretrain_hidden_layers(self, pretraining_dataloader):
        for i in range(len(self.SDAEClassifier.hidden_layers)):
            dae = AutoencoderSDAE(self.SDAEClassifier.hidden_layers[i]).to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(dae.parameters(), lr=1e-3)

            previous_layers = self.SDAEClassifier.hidden_layers[0:i]

            # TODO: think about implementing early stopping
            for epoch in range(50):
                for batch_idx, data in enumerate(pretraining_dataloader):
                    dae.train()
                    data = data.to(self.device)

                    with torch.no_grad():
                        for layer in previous_layers:
                            data = layer(data)

                    noisy_data = data.add(0.3 * torch.randn_like(data).to(self.device))

                    optimizer.zero_grad()

                    predictions = dae(noisy_data)

                    loss = criterion(predictions, data)

                    loss.backward()
                    optimizer.step()

                    print('Unsupervised Layer: {} Epoch: {} Loss: {}'.format(i, epoch, loss.item()))

    def train_classifier(self, test_dataloader, validation_dataloader):
        epochs = []
        train_losses = []
        validation_accs = []

        early_stopping = EarlyStopping('{}/{}'.format(self.model_name, self.dataset_name))

        epoch = 0
        while not early_stopping.early_stop:
            for batch_idx, (data, labels) in enumerate(test_dataloader):
                self.SDAEClassifier.train()

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                predictions = self.SDAEClassifier(data)

                loss = self.criterion(predictions, labels)

                loss.backward()
                self.optimizer.step()

                validation_acc = accuracy(self.SDAEClassifier, validation_dataloader, self.device)

                early_stopping(1 - validation_acc, self.SDAEClassifier)

                epochs.append(epoch)
                train_losses.append(loss.item())
                validation_accs.append(validation_acc)

                print('Supervised Epoch: {} Loss: {} Validation acc: {}'.format(epoch, loss.item(), validation_acc))

            epoch += 1

        early_stopping.load_checkpoint(self.SDAEClassifier)

        return epochs, train_losses, validation_accs

    def train(self, supervised_dataloader, unsupervised_dataloader, validation_dataloader=None):
        self.pretrain_hidden_layers(unsupervised_dataloader)

        classifier_epochs, classifier_train_losses, classifier_validation_accs = \
            self.train_classifier(supervised_dataloader, validation_dataloader)

        return classifier_epochs, classifier_train_losses, classifier_validation_accs

    def test(self, test_dataloader):
        return accuracy(self.SDAEClassifier, test_dataloader, self.device)
