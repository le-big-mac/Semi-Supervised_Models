import math
import torch
from torch import nn
import torch.nn.functional as F
from itertools import cycle
from Models.Model import Model
from utils.trainingutils import EarlyStopping


def bi(inits, size):
    return nn.Parameter(inits * torch.ones(size))


def wi(shape):
    return nn.Parameter(torch.randn(shape) / math.sqrt(shape[0]))


join = lambda l, u: torch.cat((l, u), 0)
labeled = lambda x, batch_size: x[:batch_size] if x is not None else x
unlabeled = lambda x, batch_size: x[batch_size:] if x is not None else x
split_lu = lambda x, batch_size: (labeled(x, batch_size), unlabeled(x, batch_size))


class encoders(nn.Module):
    def __init__(self, shapes, layer_sizes, L, device):
        super(encoders, self).__init__()
        self.W = nn.ParameterList([wi(s) for s in shapes])
        self.beta = nn.ParameterList([bi(0.0, s[1]) for s in shapes])
        self.gamma = nn.Parameter(bi(1.0, layer_sizes[-1]))
        self.batch_norm_clean_labelled = nn.ModuleList([nn.BatchNorm1d(s[1], affine=False) for s in shapes])
        self.batch_norm_clean_unlabelled = nn.ModuleList([nn.BatchNorm1d(s[1], affine=False) for s in shapes])
        self.batch_norm_noisy = nn.ModuleList([nn.BatchNorm1d(s[1], affine=False) for s in shapes])

        self.L = L
        self.device = device

    def forward(self, inputs, noise_std, training, batch_size):
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
                # softmax done in nn.CrossEntropyLoss so output from model is linear
                h = self.gamma * (z + self.beta[l-1])
            else:
                # use ReLU activation in hidden layers
                h = F.relu(z + self.beta[l-1])

            d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z, batch_size)
        d['labeled']['h'][self.L], d['unlabeled']['h'][self.L] = split_lu(h, batch_size)
        return h, d


class decoders(nn.Module):
    def __init__(self, shapes, layer_sizes, L):
        super(decoders, self).__init__()

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

        self.L = L

    def g_gauss(self, z_c, u, l):
        "gaussian denoising function proposed in the original paper"
        mu = self.a1[l] * torch.sigmoid(self.a2[l] * u + self.a3[l]) + self.a4[l] * u + self.a5[l]
        v = self.a6[l] * torch.sigmoid(self.a7[l] * u + self.a8[l]) + self.a9[l] * u + self.a10[l]

        z_est = (z_c - mu) * v + mu
        return z_est

    # Decoder
    def forward(self, y_c, corr, clean, batch_size):
        z_est = {}
        z_est_bn = {}
        for l in range(self.L, -1, -1):
            z_c = corr['unlabeled']['z'][l]
            if l == self.L:
                u = unlabeled(y_c, batch_size)
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
    def __init__(self, shapes, layer_sizes, L, device):
        super(Ladder, self).__init__()

        self.encoders = encoders(shapes, layer_sizes, L, device)
        self.decoders = decoders(shapes, layer_sizes, L)

    def forward_encoders(self, inputs, noise_std, train, batch_size):
        return self.encoders.forward(inputs, noise_std, train, batch_size)

    def forward_decoders(self, y_c, corr, clean, batch_size):
        return self.decoders.forward(y_c, corr, clean, batch_size)


class LadderNetwork(Model):
    def __init__(self, input_size, hidden_dimensions, num_classes, denoising_cost, dataset_name, device,
                 noise_std=0.3):
        super(LadderNetwork, self).__init__(dataset_name, device)

        layer_sizes = [input_size] + hidden_dimensions + [num_classes]
        shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.L = len(layer_sizes) - 1
        self.ladder = Ladder(shapes, layer_sizes, self.L, device).to(device)
        self.optimizer = torch.optim.Adam(self.ladder.parameters(), lr=1e-3)
        self.supervised_cost_function = nn.CrossEntropyLoss()
        self.unsupervised_cost_function = nn.MSELoss(reduction='mean')

        self.denoising_cost = denoising_cost
        self.noise_std = noise_std

        self.model_name = 'ladder'

    def accuracy(self, dataloader, batch_size):
        self.ladder.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs, _ = self.ladder.forward_encoders(data, 0.0, False, batch_size)

                _, predicted = torch.max(F.softmax(outputs).data, 1)
                correct += (predicted == labels).sum().item()

        return correct / len(dataloader.dataset)

    def train_ladder(self, max_epochs, supervised_dataloader, unsupervised_dataloader, validation_dataloader,
                     comparison):
        epochs = []
        train_losses = []
        validation_accs = []

        early_stopping = EarlyStopping('{}/{}.pt'.format(self.model_name, self.dataset_name))

        for epoch in range(max_epochs):
            if early_stopping.early_stop:
                break

            for batch_idx, (labelled_data, unlabelled_data) in enumerate(
                    zip(cycle(supervised_dataloader), unsupervised_dataloader)):
                self.ladder.train()

                self.optimizer.zero_grad()

                labelled_images, labels = labelled_data
                labelled_images = labelled_images.to(self.device)
                labels = labels.to(self.device)

                unlabelled_images, _ = unlabelled_data
                unlabelled_images = unlabelled_images.to(self.device)

                inputs = torch.cat((labelled_images, unlabelled_images), 0)

                batch_size = labelled_images.size(0)

                y_c, corr = self.ladder.forward_encoders(inputs, self.noise_std, True, batch_size)
                y, clean = self.ladder.forward_encoders(inputs, 0.0, True, batch_size)

                z_est_bn = self.ladder.forward_decoders(F.softmax(y_c), corr, clean, batch_size)

                cost = self.supervised_cost_function.forward(labeled(y_c, batch_size), labels)

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

                if comparison:
                    val_acc = self.accuracy(validation_dataloader, 0)

                    epochs.append(epoch)
                    train_losses.append(loss.item())
                    validation_accs.append(val_acc)

                    print('Epoch: {} Supervised Loss: {} Unsupervised Loss: {} Validation Accuracy: {}'
                          .format(epoch, cost.item(), u_cost.item(), val_acc))

            val = self.accuracy(validation_dataloader, 0)

            print('Epoch: {} Validation Accuracy: {}'.format(epoch, val))

            early_stopping(1 - val, self.ladder)

        if early_stopping.early_stop:
            early_stopping.load_checkpoint(self.ladder)

        return epochs, train_losses, validation_accs

    def train_model(self, max_epochs, dataloaders, comparison=False):
        unsupervised_dataloader, supervised_dataloader, validation_dataloader = dataloaders

        epochs, losses, validation_accs = self.train_ladder(max_epochs, supervised_dataloader, unsupervised_dataloader,
                                                            validation_dataloader, comparison)

        return epochs, losses, validation_accs

    def test_model(self, test_dataloader):
        return self.accuracy(test_dataloader, 0)

    def classify(self, data):
        self.ladder.eval()

        return self.forward(data)

    def forward(self, data):
        y, _ = self.ladder.forward_encoders(data, 0.0, False, 0)

        return y
