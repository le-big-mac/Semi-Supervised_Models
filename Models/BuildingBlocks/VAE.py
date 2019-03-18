import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, latent_dim):
        super(Encoder, self).__init__()

        dims = [input_size] + hidden_dimensions

        hidden_layers = [
            nn.Sequential(
                nn.Linear(dims[i-1], dims[i]),
                nn.ReLU(),
            )
            for i in range(1, len(dims))
        ]

        self.layers = nn.ModuleList(hidden_layers)

        self.mu = nn.Linear(hidden_dimensions[-1], latent_dim)
        # ReLU as variance has to be greater than 0
        # TODO: this is not true, check results without
        self.logvar = nn.Sequential(
            nn.Linear(hidden_dimensions[-1], latent_dim),
            nn.ReLU(),
        )

    def encode(self, x):
        for layer in self.layers:
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
    def __init__(self, input_size, hidden_dimensions, latent_dim, output_activation):
        super(Decoder, self).__init__()

        dims = hidden_dimensions + [latent_dim]
        dims = dims[::-1]

        hidden_layers = [
            nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
            ) for i in range(0, len(dims)-1)
        ]

        out = nn.Sequential(
            nn.Linear(hidden_dimensions[-1], input_size),
            output_activation,
        )

        self.layers = nn.ModuleList(hidden_layers + [out])

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)

        return z


class VAE(nn.Module):
    def __init__(self, input_size, hidden_dimensions, latent_dim, output_activation):
        super(VAE, self).__init__()

        self.encoder = Encoder(input_size, hidden_dimensions, latent_dim)
        self.decoder = Decoder(input_size, hidden_dimensions, latent_dim, output_activation)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)

        out = self.decoder(z)

        return out, mu, logvar
