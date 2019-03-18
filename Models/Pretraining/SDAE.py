import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import accuracy
from Models.Autoencoders.SingleLayerAutoencoder import Encoder, Autoencoder


class PretrainedSDAE(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation):
        super(PretrainedSDAE, self).__init__()

        dims = [input_size] + hidden_dimensions

        layers = [Encoder(input_size=dims[i - 1], layer_size=dims[i], activation=activation)
                  for i in range(1, len(dims))]

        self.hidden_layers = nn.ModuleList(layers)
        self.classification_layer = nn.Linear(layers[-1].layer.out_features, num_classes)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        return self.classification_layer(x)


class SDAE:
    def __init__(self, input_size, hidden_dimensions, num_classes, activation, device):
        self.PretrainedSDAE = self.PretrainedSDAE(input_size, hidden_dimensions, num_classes, activation).to(device)
        self.optimizer = torch.optim.Adam(self.PretrainedSDAE.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

        self.state_path = 'Models/state/sdae.pt'
        torch.save(self.PretrainedSDAE.state_dict(), self.state_path)

        self.device = device

    def pretrain_hidden_layers(self, dataloader):

        for i in range(len(self.PretrainedSDAE.hidden_layers)):
            decoder = Autoencoder(self.PretrainedSDAE.hidden_layers[i]).to(self.device)
            criterion = nn.MSELoss(reduction="sum")
            optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

            previous_layers = self.PretrainedSDAE.hidden_layers[0:i]

            for epoch in range(50):
                layer_loss = self.train_DAE_one_epoch(previous_layers, decoder, dataloader, criterion, optimizer)
                if epoch % 10 == 0:
                    print('Epoch: {} Layer loss: {}'.format(epoch, layer_loss))

    def train_DAE_one_epoch(self, previous_layers, dae, dataloader, criterion, optimizer):
        dae.train()
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data = data.to(self.device)

            with torch.no_grad():
                for layer in previous_layers:
                    data = layer(data)

            noisy_data = data.add(0.2*torch.randn_like(data).to(self.device))

            optimizer.zero_grad()

            predictions = dae(noisy_data)

            loss = criterion(predictions, data)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        return train_loss/len(dataloader.dataset)

    def supervised_train_one_epoch(self, epoch, dataloader):
        self.PretrainedSDAE.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.PretrainedSDAE(data)

            loss = self.criterion(predictions, labels)

            train_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        if epoch % 10 == 0:
            print("Epoch: {} Supervised Loss: {}".format(epoch, train_loss/len(dataloader.dataset)))

        return train_loss/len(dataloader.dataset)

    def reset_model(self):
        self.PretrainedSDAE.load_state_dict(torch.load(self.state_path))

        self.optimizer = torch.optim.Adam(self.PretrainedSDAE.parameters(), lr=1e-3)

    def full_train(self, combined_dataset, train_dataset, validation_dataset):
        self.reset_model()
        pretraining_dataloader = DataLoader(dataset=combined_dataset, batch_size=1000, shuffle=True)

        self.pretrain_hidden_layers(pretraining_dataloader)

        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(50):

            self.supervised_train_one_epoch(epoch, supervised_dataloader)
            print('Epoch: {} Validation Acc: {}'.format(epoch, accuracy.accuracy(self.PretrainedSDAE,
                                                                                 validation_dataloader, self.device)))

    def full_test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return accuracy.accuracy(self.PretrainedSDAE, test_dataloader, self.device)
