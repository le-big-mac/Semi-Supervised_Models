from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation):
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
        self.classification_layer = nn.Linear(hidden_dimensions[-1], num_classes)

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)

        return self.classification_layer(x)


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, latent_dim, activation):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(input_size, hidden_dimensions, latent_dim, activation)

        dims = hidden_dimensions + [latent_dim]

        layers = [
            nn.Sequential(
                nn.Linear(dims[i], dims[i-1]),
                activation,
            )
            for i in range(len(dims)-1, 0, -1)
        ]

        self.fc_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dimensions[0], input_size)

    def forward(self, x):
        h = self.encoder(x)

        for layer in self.fc_layers:
            h = layer(h)

        return self.output_layer(h)
