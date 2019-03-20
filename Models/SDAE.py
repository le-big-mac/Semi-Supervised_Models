import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.accuracy import accuracy
from Models.BuildingBlocks import Encoder, AutoencoderSDAE
from Models.Model import Model


class SDAE(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation):
        super(SDAE, self).__init__()

        dims = [input_size] + hidden_dimensions

        layers = [Encoder(dims[i], [], dims[i+1], activation)
                  for i in range(0, len(dims)-1)]

        self.hidden_layers = nn.ModuleList(layers)
        self.classification_layer = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        return self.classification_layer(x)


class SDAENetwork(Model):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation, device):
        super(SDAENetwork, self).__init__(device)

        self.SDAE = SDAE(input_size, hidden_dimensions, num_classes, activation).to(device)
        self.optimizer = torch.optim.Adam(self.SDAE.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def pretrain_hidden_layers(self, pretraining_dataloader):
        # TODO: take in dataset and then redo dataset each iteration (faster)
        for i in range(len(self.SDAE.hidden_layers)):
            dae = AutoencoderSDAE(self.SDAE.hidden_layers[i]).to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(dae.parameters(), lr=1e-3)

            previous_layers = self.SDAE.hidden_layers[0:i]

            for epoch in range(50):
                self.train_DAE_one_epoch(previous_layers, dae, pretraining_dataloader, criterion, optimizer)

    def train_DAE_one_epoch(self, previous_layers, dae, dataloader, criterion, optimizer):
        dae.train()

        for batch_idx, data in enumerate(dataloader):
            data = data.to(self.device)

            with torch.no_grad():
                for layer in previous_layers:
                    data = layer(data)

            noisy_data = data.add(0.2*torch.randn_like(data).to(self.device))

            optimizer.zero_grad()

            predictions = dae(noisy_data)

            loss = criterion(predictions, data)

            loss.backward()
            optimizer.step()

            print(loss.item())

    def train_classifier_one_epoch(self, epoch, dataloader, validation_dataloader):
        self.SDAE.train()

        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.SDAE(data)

            loss = self.criterion(predictions, labels)

            loss.backward()
            self.optimizer.step()

            print('Epoch: {} Loss: {} Validation accuracy: {}'
                  .format(epoch, loss.item(), accuracy(self.SDAE, validation_dataloader, self.device)))

    def train(self, unsupervised_dataset, train_dataset, validation_dataset):
        pretraining_dataloader = DataLoader(dataset=unsupervised_dataset, batch_size=1000, shuffle=True)

        self.pretrain_hidden_layers(pretraining_dataloader)

        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(50):
            self.train_classifier_one_epoch(epoch, supervised_dataloader, validation_dataloader)

    def test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return accuracy(self.SDAE, test_dataloader, self.device)
