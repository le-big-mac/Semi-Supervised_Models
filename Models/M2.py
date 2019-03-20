import torch
from torch import nn
from torch.nn import functional as F
from itertools import cycle
from torch.utils.data import DataLoader
from Models.BuildingBlocks import VariationalEncoder, Decoder, Classifier
from Models.Model import Model

# -----------------------------------------------------------------------
# Implementation of Kingma M2 semi-supervised variational autoencoder
# -----------------------------------------------------------------------


class VAE_M2(nn.Module):
    def __init__(self, input_size, hidden_dimensions_encoder, hidden_dimensions_decoder, latent_dim, num_classes,
                 output_activation):
        super(VAE_M2, self).__init__()

        self.encoder = VariationalEncoder(input_size + num_classes, hidden_dimensions_encoder, latent_dim)
        self.decoder = Decoder(input_size, hidden_dimensions_decoder, latent_dim + num_classes, output_activation)

    def forward(self, x, y):
        z, mu, logvar = self.encoder(torch.cat((x, y), dim=1))

        out = self.decoder(torch.cat((z, y), dim=1))

        return out, mu, logvar


class M2(nn.Module):
    def __init__(self, input_size, hidden_dimensions_VAE, hidden_dimensions_clas, latent_dim, num_classes,
                 output_activation):
        super(M2, self).__init__()

        self.VAE = VAE_M2(input_size, hidden_dimensions_VAE, hidden_dimensions_VAE, latent_dim, num_classes,
                          output_activation)
        self.Classifier = Classifier(input_size, hidden_dimensions_clas, num_classes)

    def classify(self, x):
        return self.Classifier(x)

    def forward(self, x, y):
        return self.VAE(x, y)


class M2Runner(Model):
    def __init__(self, input_size, hidden_dimensions_VAE, hidden_dimensions_clas, latent_dim, num_classes, activation,
               device):
        super(M2Runner, self).__init__(device)

        self.M2 = M2(input_size, hidden_dimensions_VAE, hidden_dimensions_clas, latent_dim,
                     num_classes, activation).to(device)
        # change this to something more applicable with softmax
        self.optimizer = torch.optim.Adam(self.M2.parameters(), lr=1e-3)
        self.num_classes = num_classes

    def onehot(self, labels):
        labels = labels.unsqueeze(1)

        y = torch.zeros(labels.size(0), self.num_classes).to(self.device)
        y = y.scatter(1, labels, 1)

        return y

    def minus_L(self, x, recons, mu, logvar, y):
        # KL divergence between two normal distributions (N(0, 1) and parameterized)
        KLD = 0.5*torch.sum(logvar.exp() + mu.pow(2) - logvar - 1, dim=1)

        # reconstruction error (use BCE because we normalize input data to [0, 1] and sigmoid output)
        likelihood = -F.binary_cross_entropy(recons, x, reduction='none').sum(dim=1)

        # prior over y (commented out because a uniform prior results in a constant for all labels)
        # prior_y = log_standard_categorical(y)

        return likelihood - KLD

    def log_standard_categorical(self, y):
        # this is useless when the data is uniformly distributed as it returns a constant
        prior = torch.ones((y.size(0), self.num_classes), requires_grad=False)

        return -F.cross_entropy(prior, y, reduction='none')

    def make_labels(self, batch_size):
        labels = []
        for i in range(self.num_classes):
            labels.append(i * torch.ones(batch_size).to(self.device).long())

        labels = torch.cat(labels)

        return labels

    def minus_U(self, x, pred_y):
        # gives probability for each label
        logits = F.softmax(pred_y)

        y = self.make_labels(x.size(0))
        y_onehot = self.onehot(y)
        x = x.repeat(self.num_classes, 1)

        recons, mu, logvar = self.M2(x, y_onehot)

        minus_L = self.minus_L(x, recons, mu, logvar, y)
        minus_L = minus_L.view_as(logits.t()).t()

        minus_L = (logits * minus_L).sum(dim=1)

        H = self.H(logits)

        minus_U = H + minus_L

        return minus_U.mean()

    def H(self, logits):
        return -torch.sum(logits * torch.log(logits + 1e-8), dim=1)

    def elbo(self, x, y=None):
        if y is not None:
            recons, mu, logvar = self.M2(x, self.onehot(y))

            return -self.minus_L(x, recons, mu, logvar, y).mean()

        else:
            pred_y = self.M2.classify(x)

            return -self.minus_U(x, pred_y)

    def train_one_epoch(self, epoch, labelled_loader, unlabelled_loader, validation_loader, alpha):

        for batch_idx, (labelled_data, unlabelled_data) in enumerate(zip(cycle(labelled_loader), unlabelled_loader)):
            self.M2.train()
            self.optimizer.zero_grad()

            labelled_images, labels = labelled_data
            labelled_images = labelled_images.float().to(self.device)
            labels = labels.to(self.device)

            unlabelled_images = unlabelled_data.float().to(self.device)

            labelled_predictions = self.M2.classify(labelled_images)
            labelled_loss = F.cross_entropy(labelled_predictions, labels)

            # labelled images ELBO
            L = self.elbo(labelled_images, y=labels)

            U = self.elbo(unlabelled_images)

            loss = L + U + alpha*labelled_loss

            loss.backward()
            self.optimizer.step()

            # print('Epoch: {} Classification Loss: {} Unlabelled Loss: {} Labelled Loss: {} Validation Accuracy: {}'
            #       .format(epoch, labelled_loss.item(), U.item(), L.item(), self.evaluate(validation_loader)))

        print('Epoch: {} Validation Accuracy: {}'.format(epoch, self.evaluate(validation_loader)))

    def evaluate(self, dataloader):
        self.M2.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.float().to(self.device)
                labels = labels.to(self.device)

                outputs = self.M2.classify(data)

                _, predicted = torch.max(F.softmax(outputs).data, 1)
                correct += (predicted == labels).sum().item()

        return correct / len(dataloader.dataset)

    def train(self, unsupervised_dataset, supervised_train_dataset, validation_dataset, alpha):

        labelled_loader = DataLoader(supervised_train_dataset, batch_size=100, shuffle=True)
        unlabelled_loader = DataLoader(unsupervised_dataset, batch_size=100, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=validation_dataset.__len__())

        print('Prior Accuracy: {}'.format(self.evaluate(validation_loader)))
        for epoch in range(100):
            self.train_one_epoch(epoch, labelled_loader, unlabelled_loader, validation_loader, alpha)

    def test(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__())

        return self.evaluate(test_loader)
