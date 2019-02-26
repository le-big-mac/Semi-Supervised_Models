import torch
from torch import nn
from torch import functional as F
from Models.VAE.VAE import VAE


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

        self.VAE = VAE(input_size, latent_dim, latent_dim+num_classes, hidden_dimensions_VAE, activation).to(device)
        self.Classifier = Classifier(input_size, hidden_dimensions_clas, num_classes, activation).to(device)
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
