import math
import torch
from torch import nn
import torch.nn.functional as F
from itertools import cycle
from torch.utils.data import DataLoader
from Models.Model import Model


def bi(inits, size):
    return nn.Parameter(inits * torch.ones(size))


def wi(shape):
    return nn.Parameter(torch.randn(shape) / math.sqrt(shape[0]))


def join(l, u):
    return torch.cat((l, u), 0)


def labeled(x, batch_size):
    return x[:batch_size] if x is not None else x


def unlabeled(x, batch_size):
    return x[batch_size:] if x is not None else x


def split_lu(x, batch_size):
    return labeled(x, batch_size), unlabeled(x, batch_size)


class Encoders(nn.Module):
    def __init__(self, layer_sizes, L, shapes, device):
        super(Encoders, self).__init__()
        self.L = L
        self.W = nn.ParameterList([wi(s) for s in shapes])
        self.beta = nn.ParameterList([bi(0.0, s[1]) for s in shapes])
        self.gamma = nn.Parameter(bi(1.0, layer_sizes[-1]))
        self.batch_norm_clean_labelled = nn.ModuleList([nn.BatchNorm1d(s[1], affine=False) for s in shapes])
        self.batch_norm_clean_unlabelled = nn.ModuleList([nn.BatchNorm1d(s[1], affine=False) for s in shapes])
        self.batch_norm_noisy = nn.ModuleList([nn.BatchNorm1d(s[1], affine=False) for s in shapes])

        self.device = device

    def forward(self, inputs, noise_std, batch_size, training):
        h = inputs + noise_std * torch.randn_like(inputs).to(self.device)  # add noise to input
        d = {}  # to store the pre-activation, activation, mean and variance for each layer
        # The data for labeled and unlabeled examples are stored separately
        d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h, batch_size)
        for l in range(1, self.L+1):
            # print("Layer ", l, ": ", layer_sizes[l-1], " -> ", layer_sizes[l])

            d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h, batch_size)
            z_pre = torch.mm(h, self.W[l-1])  # pre-activation

            if training:
                z_pre_l, z_pre_u = split_lu(z_pre, batch_size)  # split labeled and unlabeled examples
                m = z_pre_u.mean(dim=0)
                v = z_pre_u.var(dim=0)
                d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
                # Training batch normalization
                # batch normalization for labeled and unlabeled examples is performed separately
                if noise_std > 0:
                    # Corrupted encoder
                    # batch normalization + noise
                    z = join(self.batch_norm_noisy[l-1](z_pre_l), self.batch_norm_noisy[l-1](z_pre_u))
                    z += noise_std * torch.randn_like(z).to(self.device)
                else:
                    # Clean encoder
                    # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                    z = join(self.batch_norm_clean_labelled[l-1](z_pre_l),
                             self.batch_norm_clean_unlabelled[l-1](z_pre_u))

            else:
                # Evaluation batch normalization
                z = self.batch_norm_clean_labelled[l-1](z_pre)

            if l == self.L:
                # use softmax activation in output layer
                # softmax done outside to allow crossentropyloss
                h = self.gamma * (z + self.beta[l-1])
            else:
                # use ReLU activation in hidden layers
                h = F.relu(z + self.beta[l-1])

            d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z, batch_size)
        d['labeled']['h'][self.L], d['unlabeled']['h'][self.L] = split_lu(h, batch_size)
        return h, d


class Decoders(nn.Module):
    def __init__(self, layer_sizes, L, shapes):
        super(Decoders, self).__init__()
        self.L = L

        self.V = nn.ParameterList([wi(s[::-1]) for s in shapes])

        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(size, affine=False) for size in layer_sizes])

        self.a1 = nn.ParameterList([bi(0., size) for size in layer_sizes])
        self.a2 = nn.ParameterList([bi(1., size) for size in layer_sizes])
        self.a3 = nn.ParameterList([bi(0., size) for size in layer_sizes])
        self.a4 = nn.ParameterList([bi(0., size) for size in layer_sizes])
        self.a5 = nn.ParameterList([bi(0., size) for size in layer_sizes])

        self.a6 = nn.ParameterList([bi(0., size) for size in layer_sizes])
        self.a7 = nn.ParameterList([bi(1., size) for size in layer_sizes])
        self.a8 = nn.ParameterList([bi(0., size) for size in layer_sizes])
        self.a9 = nn.ParameterList([bi(0., size) for size in layer_sizes])
        self.a10 = nn.ParameterList([bi(0., size) for size in layer_sizes])

    def g_gauss(self, z_c, u, l):
        "gaussian denoising function proposed in the original paper"
        mu = self.a1[l] * torch.sigmoid(self.a2[l] * u + self.a3[l]) + self.a4[l] * u + self.a5[l]
        v = self.a6[l] * torch.sigmoid(self.a7[l] * u + self.a8[l]) + self.a9[l] * u + self.a10[l]

        z_est = (z_c - mu) * v + mu
        return z_est

    # Decoder
    def forward(self, y_c, corr, clean):
        z_est = {}
        z_est_bn = {}
        for l in range(self.L, -1, -1):
            z_c = corr['unlabeled']['z'][l]
            if l == self.L:
                u = unlabeled(y_c)
            else:
                u = torch.mm(z_est[l+1], self.V[l])
            u = self.batch_norm[l](u)
            z_est[l] = self.g_gauss(z_c, u, l)

            if l > 0:
                m = clean['unlabeled']['m'][l]
                v = clean['unlabeled']['v'][l]
                z_est_bn[l] = (z_est[l] - m) / torch.sqrt(v + 1e-10)
            else:
                z_est_bn[l] = z_est[l]

        return z_est_bn


class Ladder(nn.Module):
    def __init__(self, layer_sizes, L, shapes, device):
        super(Ladder, self).__init__()

        self.encoders = Encoders(layer_sizes, L, shapes, device)
        self.decoders = Decoders(layer_sizes, L, shapes)

    def forward_encoders(self, inputs, noise_std, batch_size, train):
        return self.encoders.forward(inputs, noise_std, batch_size, train)

    def forward_decoders(self, y_c, corr, clean):
        return self.decoders.forward(y_c, corr, clean)


class LadderNetwork(Model):
    def __init__(self, input_size, hidden_dimensions, num_classes, denoising_cost, device, noise_std=0.3):
        super(LadderNetwork, self).__init__(device)

        layer_sizes = [input_size] + hidden_dimensions + [num_classes]
        L = len(layer_sizes) - 1 # number of layers
        shapes = list(zip(layer_sizes[:-1], layer_sizes[1:])) # shapes of linear layers

        self.denoising_cost = denoising_cost
        self.noise_std = noise_std

        self.ladder = Ladder(layer_sizes, L, shapes, device)
        self.optimizer = torch.optim.Adam(self.ladder.parameters(), lr=1e-3)
        self.supervised_cost_function = nn.CrossEntropyLoss()
        self.unsupervised_cost_function = nn.MSELoss(reduction="mean")

    def train_one_epoch(self, epoch, labelled_loader, unlabelled_loader, validation_loader):
        for batch_idx, (labelled_data, unlabelled_data) in enumerate(zip(cycle(labelled_loader), unlabelled_loader)):
            self.ladder.train()

            self.optimizer.zero_grad()

            labelled_images, labels = labelled_data
            labelled_images = labelled_images.float().to(self.device)
            labels = labels.to(self.device)

            unlabelled_images = unlabelled_data.float().to(self.device)

            inputs = torch.cat((labelled_images, unlabelled_images), 0)

            y_c, corr = self.ladder.forward_encoders(inputs, self.noise_std, labelled_images.size(0), True)
            y, clean = self.ladder.forward_encoders(inputs, 0.0, labelled_images.size(0), True)

            z_est_bn = self.ladder.forward_decoders(F.softmax(y_c), corr, clean)

            cost = self.supervised_cost_function.forward(labeled(y_c), labels)

            zs = clean['unlabeled']['z']

            u_cost = 0
            for l in range(self.L, -1, -1):
                # print('z_est', z_est_bn[l][0])
                # print('z', zs[l][0])
                # print(unsupervised_cost_function.forward(z_est_bn[l][0], zs[l][0]))
                u_cost += self.unsupervised_cost_function.forward(z_est_bn[l], zs[l]) * self.denoising_cost[l]

            loss = cost + u_cost

            loss.backward()
            self.optimizer.step()

            print('Epoch: {} Supervised Cost: {} Unsupervised Cost: {} Validation Accuracy: {}'.format(
                epoch, cost.item(), u_cost.item(), self.evaluate(validation_loader)
            ))

    def evaluate(self, dataloader):
        self.ladder.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.float().to(self.device)
                labels = labels.to(self.device)

                outputs, _ = self.ladder.forward_encoders(data, 0.0, False)

                _, predicted = torch.max(F.softmax(outputs).data, 1)
                correct += (predicted == labels).sum().item()

        return correct / len(dataloader.dataset)

    def train(self, unsupervised_dataset, supervised_dataset, validation_dataset):
        labelled_loader = DataLoader(supervised_dataset, batch_size=100, shuffle=True)
        unlabelled_loader = DataLoader(unsupervised_dataset, batch_size=100, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(50):
            self.train_one_epoch(epoch, labelled_loader, unlabelled_loader, validation_loader)

    def test(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__())

        return self.evaluate(test_loader)
