import torch
from torch import nn
from Models.BuildingBlocks import Classifier
from Models.Model import Model
from utils.trainingutils import accuracy, EarlyStopping


class SimpleNetwork(Model):
    def __init__(self, input_size, hidden_dimensions, num_classes, dataset_name, device):
        super(SimpleNetwork, self).__init__(dataset_name, device)

        self.Classifier = Classifier(input_size, hidden_dimensions, num_classes).to(device)
        self.optimizer = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        self.model_name = 'simple'

    def train_classifier(self, train_dataloader, validation_dataloader):
        epochs = []
        train_losses = []
        validation_accs = []

        early_stopping = EarlyStopping('{}/{}'.format(self.model_name, self.dataset_name))

        epoch = 0
        while not early_stopping.early_stop:
            for batch_idx, (data, labels) in enumerate(train_dataloader):
                self.Classifier.train()

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                preds = self.Classifier(data)

                loss = self.criterion(preds, labels)

                loss.backward()
                self.optimizer.step()

                validation_acc = accuracy(self.Classifier, validation_dataloader, self.device)

                early_stopping(1-validation_acc, self.Classifier)

                epochs.append(epoch)
                train_losses.append(loss.item())
                validation_accs.append(accuracy(self.Classifier, validation_dataloader, self.device))

                print('Supervised Epoch: {} Loss: {} Validation acc: {}'.format(epoch, loss.item(), validation_acc))

            epoch += 1

        early_stopping.load_checkpoint(self.Classifier)

        return epochs, train_losses, validation_accs

    def train(self, supervised_dataloader, validation_dataloader=None):
        epochs, losses, validation_accs = self.train_classifier(supervised_dataloader, validation_dataloader)

        return epochs, losses, validation_accs

    def test(self, test_dataloader):
        return accuracy(self.Classifier, test_dataloader, self.device)
