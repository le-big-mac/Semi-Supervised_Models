import torch
from torch import nn
from torch import functional as F
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(1366, 256)
        self.fc21 = nn.Linear(256, 20)
        self.fc22 = nn.Linear(256, 20)

    def encode(self, x):
        hidden = F.relu(self.fc1(x))
        return self.fc21(hidden), self.fc22(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn()
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar), mu, logvar

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(1366, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):



class M2(nn.Module):
    def __init__(self):
        super(M2, self).__init__()

        self.encoder = Encoder()
        self.fc3 = nn.Linear(20, 256)
        self.fc4 = nn.Linear(256, 1366)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        h3 = F.relu(self.fc3(z))
        out = torch.sigmoid(self.fc4(h3))
        return out, mu, logvar


class RegressionNetwork(nn.Module):
    def __init__(self, unsupervised_layers):
        super(RegressionNetwork, self).__init__()

        self.unsupervised_layers = unsupervised_layers
        self.supervised_layers = nn.Sequential(
            nn.Linear(20, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        z, _, _ = self.unsupervised_layers(x)
        return self.supervised_layers(z)