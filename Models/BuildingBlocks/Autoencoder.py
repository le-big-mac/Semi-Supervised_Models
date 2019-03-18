from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, latent_activation):
        super(Encoder, self).__init__()

        dims = [input_size] + hidden_dimensions

        hidden_layers = [
            nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
            )
            for i in range(0, len(dims)-1)
        ]

        latent = nn.Sequential(
            nn.Linear(hidden_dimensions[-1], num_classes),
            latent_activation,
        )

        self.layers = nn.ModuleList(hidden_layers + [latent])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, latent_dim, output_activation):
        super(Decoder, self).__init__()

        dims = [latent_dim] + hidden_dimensions[::-1]

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


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, latent_dim, latent_activation, output_activation):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(input_size, hidden_dimensions, latent_dim, latent_activation)
        self.decoder = Decoder(input_size, hidden_dimensions, latent_dim, output_activation)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y


class AutoencoderSDAE(nn.Module):
    def __init__(self, encoder):
        super(AutoencoderSDAE, self).__init__()

        self.encoder = encoder
        self.decoder = Decoder(encoder.in_features, [], encoder.out_features, lambda x: x)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y
