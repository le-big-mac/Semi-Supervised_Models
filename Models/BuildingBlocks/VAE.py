import torch
from torch import nn
from .Autoencoder import Decoder


class VariationalEncoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, latent_dim):
        super(VariationalEncoder, self).__init__()

        dims = [input_size] + hidden_dimensions

        layers = [
            nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
            )
            for i in range(0, len(dims)-1)
        ]

        self.hidden_layers = nn.ModuleList(layers)

        self.mu = nn.Linear(hidden_dimensions[-1], latent_dim)
        # ReLU as variance has to be greater than 0
        # TODO: this is not true, check results without
        # self.logvar = nn.Sequential(
        #     nn.Linear(hidden_dimensions[-1], latent_dim),
        #     nn.ReLU(),
        # )
        self.logvar = nn.Linear(hidden_dimensions[-1], latent_dim)

    def encode(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar), mu, logvar


class VAE(nn.Module):
    def __init__(self, input_size, hidden_dimensions, latent_dim, output_activation):
        super(VAE, self).__init__()

        self.encoder = VariationalEncoder(input_size, hidden_dimensions, latent_dim)
        self.decoder = Decoder(input_size, hidden_dimensions, latent_dim, output_activation)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)

        out = self.decoder(z)

        return out, mu, logvar
