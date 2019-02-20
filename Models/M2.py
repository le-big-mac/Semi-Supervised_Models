import torch
from torch import nn
from torch import functional as F
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dimensions, activation):
        super(Encoder, self).__init__()

        dims = [input_size] + hidden_dimensions

        layers = [
            nn.Sequential(
                nn.Linear(dims[i-1], dims[i]),
                activation,
            )
            for i in range(len(dims))
        ]

        self.fc_layers = nn.ModuleList(layers)
        self.mu = nn.Linear(hidden_dimensions[-1], latent_dim)

        # variance has to be positive
        self.logvar = nn.Sequential(
            nn.Linear(hidden_dimensions[-1], latent_dim),
            nn.ReLU(),
        )

    def encode(self, x):
        for layer in self.fc_layers:
            x = layer(x)

        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add(mu)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar), mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dimensions, num_classes, activation):
        super(Decoder, self).__init__()

        # the generative model takes in the latent_dim and the class (one-hot)
        dims = hidden_dimensions + [latent_dim + num_classes]

        layers = [
            nn.Sequential(
                nn.Linear(dims[i], dims[i - 1]),
                activation,
            ) for i in range(len(dims), 1, -1)
        ]

        self.fc_layers = nn.ModuleList(layers)
        self.out = nn.Sequential(
            nn.Linear(hidden_dimensions[0], input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)

        return self.out(x)


class VAE(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dimensions, num_classes, activation):
        super(VAE, self).__init__()

        self.encoder = Encoder(input_size, latent_dim, hidden_dimensions, activation)
        self.decoder = Decoder(input_size, latent_dim, hidden_dimensions, num_classes, activation)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)

        out = self.decoder(z)

        return out, mu, logvar


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation):
        super(Classifier, self).__init__()

        dimensions = [input_size] + hidden_dimensions

        layers = [
            nn.Sequential(
                nn.Linear(dimensions[i - 1], dimensions[i]),
                activation,
            )
            for i in range(1, len(dimensions))
        ]

        self.fc_layers = nn.ModuleList(layers)
        self.output_layer = nn.Sequential(
            nn.Linear(dimensions[-1], num_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)

        return self.output_layer(x)


class M2:
    def __init(self, input_size, hidden_dimensions_VAE, hidden_dimensions_clas, latent_dim, num_classes, activation,
               device):

        self.VAE = VAE(input_size, latent_dim, hidden_dimensions_VAE, num_classes, activation)
        self.Classifier = Classifier(input_size, hidden_dimensions_clas, num_classes, activation)
        # change this to something more applicable with softmax
        self.classification_loss = nn.CrossEntropyLoss(reduction='sum')


    def VAE_criterion(self, pred_x, x, mu, logvar):
        # KL divergence between two normal distributions (N(0, 1) and parameterized)
        KLD = 0.5*torch.sum(logvar.exp() + mu.pow(2) - logvar - 1)

        # reconstruction error (use BCE because we normalize input data to [0, 1] and sigmoid output)
        BCE = F.binary_cross_entropy(pred_x, x, reduction='sum')

        # actually not necessarily 0-1 normalised
        # BCE = nn.MSELoss(reduction='sum')(pred_x, x)

        return KLD + BCE
