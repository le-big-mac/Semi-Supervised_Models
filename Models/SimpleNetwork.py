import torch
from torch import nn
from torch.utils.data import DataLoader
from Models.BuildingBlocks.Classifier import Classifier
from utils.accuracy import accuracy


class SimpleNetwork:
    def __init__(self, input_size, hidden_dimensions, num_classes, activation, device):
        self.Classifier = Classifier(input_size, hidden_dimensions, num_classes, activation).to(device)
        self.optimizer = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        self.device = device

    def train_one_epoch(self, epoch, train_dataloader, validation_dataloader):
        self.Classifier.train()

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            preds = self.Classifier(data)

            loss = self.criterion(preds, labels)

            loss.backward()
            self.optimizer.step()

            print('Epoch: {} Loss: {} Validation accuracy: {}'
                  .format(epoch, loss.item(), accuracy(self.Classifier, validation_dataloader, self.device)))

    def train(self, supervised_dataset, validation_dataset):
        supervised_dataloader = DataLoader(dataset=supervised_dataset, batch_size=100, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(50):
            self.train_one_epoch(epoch, supervised_dataloader, validation_dataloader)

    def test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return accuracy(self.Classifier, test_dataloader, self.device)
