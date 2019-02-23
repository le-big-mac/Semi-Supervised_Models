import torch
import csv
import os
from torch import nn
from torch.utils.data import DataLoader
from utils import Accuracy, Arguments, KFoldSplits, Datasets, LoadData


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation):
        super(Classifier, self).__init__()

        dimensions = [input_size] + hidden_dimensions

        layers = [
            nn.Sequential(
                nn.Linear(dimensions[i-1], dimensions[i]),
                activation,
            )
            for i in range(1, len(dimensions))
        ]

        self.fc_layers = nn.ModuleList(layers)

        self.classification_layer = nn.Linear(dimensions[-1], num_classes)

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)

        return self.classification_layer(x)

class SimpleNetwork:
    def __init__(self, input_size, hidden_dimensions, num_classes, activation, device):
        self.Classifier = Classifier(input_size, hidden_dimensions, num_classes, activation).to(device)
        self.optimizer = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.state_path = 'Models/state/simple_mlp.pt'
        self.device = device
        torch.save(self.Classifier.state_dict(), self.state_path)

    def train_classifier_one_epoch(self, epoch, dataloader):
        self.Classifier.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            preds = self.Classifier(data)

            loss = self.criterion(preds, labels)

            train_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        print('Epoch: {} Loss: {}'.format(epoch, train_loss/len(dataloader.dataset)))

        return train_loss/len(dataloader.dataset)

    def reset_model(self):
        self.Classifier.load_state_dict(torch.load(self.state_path))

        self.optimizer = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)

    def full_train(self, train_dataset, validation_dataset):
        self.reset_model()

        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(50):

            self.train_classifier_one_epoch(epoch, supervised_dataloader)
            print('Epoch: {} Validation Acc: {}'.format(epoch, Accuracy.accuracy(self.Classifier, validation_dataloader,
                                                                                 self.device)))

    def full_test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return Accuracy.accuracy(self.Classifier, test_dataloader, self.device)
