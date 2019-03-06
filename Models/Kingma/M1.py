import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Models.Autoencoders.VAE import VAE


class Classifier(nn.Module):
    def __init__(self, latent_dim, hidden_dimensions_classifier, num_classes):
        super(Classifier, self).__init__()

        dims = [latent_dim] + hidden_dimensions_classifier

        layers = [nn.Sequential(
            nn.Linear(dims[i-1], dims[i]),
            nn.ReLU(),
        ) for i in range(1, len(dims))]

        self.layers = nn.ModuleList(layers)
        self.classification = nn.Linear(dims[-1], num_classes)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)

        return self.classification(z)


class M1:
    def __init__(self, input_size, hidden_dimensions_encoder, latent_size, hidden_dimensions_classifier,
                 num_classes, activation, device):
        self.VAE = VAE(input_size, latent_size, latent_size, hidden_dimensions_encoder, activation).to(device)
        self.VAE_optim = torch.optim.Adam(self.VAE.parameters(), lr=1e-3)
        self.Encoder = self.VAE.encoder

        self.Classifier = Classifier(latent_size, hidden_dimensions_classifier, num_classes).to(device)
        self.Classifier_criterion = nn.CrossEntropyLoss(reduction='sum')
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)

        self.device = device

        self.vae_state_path = 'Models/state/m1_vae.pt'
        self.clas_state_path = 'Models/state/m1_classifier.pt'
        torch.save(self.VAE.state_dict(), self.vae_state_path)
        torch.save(self.Classifier.state_dict(), self.clas_state_path)

    def VAE_criterion(self, pred_x, x, mu, logvar):
        # KL divergence between two normal distributions (N(0, 1) and parameterized)
        KLD = 0.5*torch.sum(logvar.exp() + mu.pow(2) - logvar - 1)

        # reconstruction error (use BCE because we normalize input data to [0, 1] and sigmoid output)
        BCE = F.binary_cross_entropy(pred_x, x, reduction='sum')

        # if not necessarily 0-1 normalised
        # BCE = nn.MSELoss(reduction='sum')(pred_x, x)

        return KLD + BCE

    def train_VAE_one_epoch(self, epoch, dataloader):
        self.VAE.train()
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data = data.to(self.device)

            self.VAE_optim.zero_grad()

            recons, mu, logvar = self.VAE(data)

            loss = self.VAE_criterion(recons, data, mu, logvar)

            train_loss += loss.item()

            loss.backward()
            self.VAE_optim.step()

        print('Epoch: {} VAE Loss: {}'.format(epoch, train_loss/len(dataloader.dataset)))

        return train_loss/len(dataloader.dataset)

    def train_classifier_one_epoch(self, epoch, dataloader):
        self.Classifier.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.Classifier_optim.zero_grad()

            with torch.no_grad():
                z, _, _ = self.Encoder(data)

            pred = self.Classifier(z)

            loss = self.Classifier_criterion(pred, labels)

            train_loss += loss.item()

            loss.backward()
            self.Classifier_optim.step()

        print('Epoch {} Loss {}:'.format(epoch, train_loss/len(dataloader.dataset)))

        return train_loss/len(dataloader.dataset)

    def supervised_test(self, dataloader):
        self.Encoder.eval()
        self.Classifier.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                z, _, _ = self.Encoder(data)
                outputs = self.Classifier(z)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        return correct / len(dataloader.dataset)

    def reset_model(self):
        self.VAE.load_state_dict(torch.load(self.vae_state_path))
        self.Classifier.load_state_dict(torch.load(self.clas_state_path))

        self.VAE_optim = torch.optim.Adam(self.VAE.parameters(), lr=1e-3)
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=1e-3)

    def full_train(self, combined_dataset, train_dataset, validation_dataset=None):
        self.reset_model()

        pretraining_dataloader = DataLoader(dataset=combined_dataset, batch_size=1000, shuffle=True)

        for epoch in range(50):
            self.train_VAE_one_epoch(epoch, pretraining_dataloader)

        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(50):
            self.train_classifier_one_epoch(epoch, supervised_dataloader)
            print('Epoch: {} Validation Acc: {}'.format(epoch, self.supervised_test(validation_dataloader)))

    def full_test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return self.supervised_test(test_dataloader)
