import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_dimensions, activation):
        super(Encoder, self).__init__()

        dims = [input_size] + hidden_dimensions

        layers = [
            nn.Sequential(
                nn.Linear(dims[i-1], dims[i]),
                activation,
            )
            for i in range(1, len(dims))
        ]

        self.fc_layers = nn.ModuleList(layers)
        self.mu = nn.Linear(hidden_dimensions[-1], latent_dim)

        # variance has to be greater than 0
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
    def __init__(self, input_size, latent_dim, hidden_dimensions, activation):
        super(Decoder, self).__init__()

        # the generative model takes in the latent_dim and the class (one-hot)
        dims = hidden_dimensions + [latent_dim]

        layers = [
            nn.Sequential(
                nn.Linear(dims[i], dims[i - 1]),
                activation,
            ) for i in range(len(dims)-1, 0, -1)
        ]

        self.fc_layers = nn.ModuleList(layers)
        self.out = nn.Sequential(
            nn.Linear(hidden_dimensions[0], input_size),
            nn.Sigmoid(),
        )

    def forward(self, z):
        for layer in self.fc_layers:
            z = layer(z)

        return self.out(z)


class VAE(nn.Module):
    def __init__(self, input_size, latent_dim_encoder, latent_dim_decoder, hidden_dimensions, activation):
        super(VAE, self).__init__()

        # encoder and decoder may have different latent dim e.g. M2
        self.encoder = Encoder(input_size, latent_dim_encoder, hidden_dimensions, activation)
        self.decoder = Decoder(input_size, latent_dim_decoder, hidden_dimensions, activation)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)

        out = self.decoder(z)

        return out, mu, logvar

