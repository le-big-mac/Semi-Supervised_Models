from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes):
        super(Classifier, self).__init__()

        dims = [input_size] + hidden_dimensions

        layers = [
            nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
            )
            for i in range(0, len(dims)-1)
        ]

        self.hidden_layers = nn.ModuleList(layers)

        self.out = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        return self.out(x)
