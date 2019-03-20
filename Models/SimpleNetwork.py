import torch
from torch import nn
from torch.utils.data import DataLoader
from Models.BuildingBlocks import Classifier
from Models.Model import Model
from utils.trainingutils import accuracy, EarlyStopping


class SimpleNetwork(Model):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation, device):
        super(SimpleNetwork, self).__init__(device)

        self.Classifier = Classifier(input_size, hidden_dimensions, num_classes).to(device)
        self.optimizer = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        self.model_name = 'simple_network'

    def train_one_epoch(self, train_dataloader):
        self.Classifier.train()

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            preds = self.Classifier(data)

            loss = self.criterion(preds, labels)

            loss.backward()
            self.optimizer.step()

            return loss.item()

    def train(self, dataset_name, supervised_dataset, batch_size, validation_dataset):
        supervised_dataloader = DataLoader(dataset=supervised_dataset, batch_size=100, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        early_stopping = EarlyStopping('{}/{}'.format(self.model_name, dataset_name))

        epochs = []
        losses = []
        validation_accs = []

        epoch = 0
        while not early_stopping.early_stop:
            loss = self.train_one_epoch(supervised_dataloader)
            validation_acc = accuracy(self.Classifier, validation_dataloader, self.device)

            early_stopping(1-validation_acc, self.Classifier)

            epochs.append(epoch)
            losses.append(loss)
            validation_accs.append(validation_acc)

            epoch += 1

        return epochs, losses, validation_accs

    def test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return accuracy(self.Classifier, test_dataloader, self.device)
