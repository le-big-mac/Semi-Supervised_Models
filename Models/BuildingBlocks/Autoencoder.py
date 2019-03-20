from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, latent_activation):
        super(Encoder, self).__init__()

        dims = [input_size] + hidden_dimensions

        layers = [
            nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
            )
            for i in range(0, len(dims)-1)
        ]

        self.hidden_layers = nn.ModuleList(layers)

        self.latent = nn.Linear(dims[-1], num_classes)
        self.latent_activation = latent_activation

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        return self.latent_activation(self.latent(x))


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, latent_dim, output_activation):
        super(Decoder, self).__init__()

        dims = [latent_dim] + hidden_dimensions[::-1]

        layers = [
            nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
            ) for i in range(0, len(dims)-1)
        ]

        self.hidden_layers = nn.ModuleList(layers)

        self.out = nn.Linear(dims[-1], input_size)
        self.output_activation = output_activation

    def forward(self, z):
        for layer in self.hidden_layers:
            z = layer(z)

        return self.output_activation(self.out(z))


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
