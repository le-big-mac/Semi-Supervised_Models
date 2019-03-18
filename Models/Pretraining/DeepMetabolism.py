import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import accuracy
from Models.Autoencoders.MultilayerAutoencoder import Encoder, Autoencoder


class DeepMetabolism:
    def __init__(self, input_size, hidden_dimensions, num_classes, activation, device):
        self.Autoencoder = Autoencoder(input_size, hidden_dimensions, num_classes, activation).to(device)
        self.Autoencoder_optim = torch.optim.Adam(self.Autoencoder.parameters(), lr=1e-3)
        self.Autoencoder_criterion = nn.MSELoss(reduction='sum')

        self.Classifier = self.Autoencoder.encoder
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)
        self.Classifier_criterion = nn.CrossEntropyLoss(reduction='sum')

        self.state_path = 'Models/state/deep_metabolism.pt'
        torch.save(self.Autoencoder.state_dict(), self.state_path)

        self.device = device

    def unsupervised_train_one_epoch(self, dataloader):
        self.Autoencoder.train()
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data = data.to(self.device)

            self.Autoencoder_optim.zero_grad()

            recons = self.Autoencoder(data)

            loss = self.Autoencoder_criterion(recons, data)

            train_loss += loss.item()

            loss.backward()
            self.Autoencoder_optim.step()

        print('Unsupervised Loss: {}'.format(train_loss/len(dataloader.dataset)))

        return train_loss/len(dataloader.dataset)

    def train_classifier_one_epoch(self, epoch, dataloader):
        self.Classifier.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.Classifier_optim.zero_grad()

            preds = self.Classifier(data)

            loss = self.Classifier_criterion(preds, labels)

            train_loss += loss.item()

            loss.backward()
            self.Classifier_optim.step()

        print('Epoch: {} Loss: {}'.format(epoch, train_loss/len(dataloader.dataset)))

        return train_loss/len(dataloader.dataset)

    def reset_model(self):
        self.Autoencoder.load_state_dict(torch.load(self.state_path))
        self.Classifier = self.Autoencoder.encoder

        self.Autoencoder_optim = torch.optim.Adam(self.Autoencoder.parameters(), lr=1e-3)
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)

    def full_train(self, combined_dataset, train_dataset, validation_dataset=None):
        self.reset_model()

        pretraining_dataloader = DataLoader(dataset=combined_dataset, batch_size=1000, shuffle=True)

        for epoch in range(50):
            self.unsupervised_train_one_epoch(pretraining_dataloader)

        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(50):

            self.train_classifier_one_epoch(epoch, supervised_dataloader)
            print('Epoch: {} Validation Acc: {}'.format(epoch, accuracy.accuracy(self.Classifier, validation_dataloader,
                                                                                 self.device)))

    def full_test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return accuracy.accuracy(self.Classifier, test_dataloader, self.device)
