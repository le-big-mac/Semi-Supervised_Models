from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, layer_size, activation):
        super(Encoder, self).__init__()

        self.layer = nn.Linear(input_size, layer_size)
        self.activation = activation

    def forward(self, x):
        h = self.layer(x)
        if self.activation:
            h = self.activation(h)

        return h


class Autoencoder(nn.Module):
    def __init__(self, encoder):
        super(Autoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = nn.Linear(encoder.layer.out_features, encoder.layer.in_features)

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)
