import math
import torch
from torch import nn
from torch.nn import Parameter, Module
import torch.nn.functional as F
from itertools import cycle
from utils.LoadData import load_MNIST_data
from torch.utils.data import DataLoader

layer_sizes = [784, 1000, 500, 250, 250, 250, 10]

L = len(layer_sizes) - 1  # number of layers

num_examples = 60000
num_epochs = 150
num_labeled = 100

starter_learning_rate = 0.02

decay_after = 15  # epoch after which to begin learning rate decay

batch_size = 100
num_iter = (num_examples/batch_size) * num_epochs  # number of loop iterations


def bi(inits, size):
    return Parameter(inits * torch.ones(size))


def wi(shape):
    return Parameter(torch.randn(shape) / math.sqrt(shape[0]))


shapes = list(zip(layer_sizes[:-1], layer_sizes[1:])) # shapes of linear layers

noise_std = 0.3  # scaling factor for noise used in corrupted encoder

# hyperparameters that denote the importance of each layer
denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]

join = lambda l, u: torch.cat((l, u), 0)
labeled = lambda x: x[:batch_size] if x is not None else x
unlabeled = lambda x: x[batch_size:] if x is not None else x
split_lu = lambda x: (labeled(x), unlabeled(x))


class encoders(Module):
    def __init__(self):
        super(encoders, self).__init__()
        self.W = nn.ParameterList([wi(s) for s in shapes])
        self.beta = nn.ParameterList([bi(0.0, s[1]) for s in shapes])
        self.gamma = nn.Parameter(bi(1.0, layer_sizes[-1]))
        self.batch_norm_clean_labelled = nn.ModuleList([nn.BatchNorm1d(s[1], affine=False) for s in shapes])
        self.batch_norm_clean_unlabelled = nn.ModuleList([nn.BatchNorm1d(s[1], affine=False) for s in shapes])
        self.batch_norm_noisy = nn.ModuleList([nn.BatchNorm1d(s[1], affine=False) for s in shapes])

    def forward(self, inputs, noise_std, training):
        h = inputs + torch.randn_like(inputs) * noise_std  # add noise to input
        d = {}  # to store the pre-activation, activation, mean and variance for each layer
        # The data for labeled and unlabeled examples are stored separately
        d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
        for l in range(1, L+1):
            # print("Layer ", l, ": ", layer_sizes[l-1], " -> ", layer_sizes[l])

            d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
            z_pre = torch.mm(h, self.W[l-1])  # pre-activation

            if training:
                z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples
                m = z_pre_u.mean(dim=0)
                v = z_pre_u.var(dim=0)
                d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
                # Training batch normalization
                # batch normalization for labeled and unlabeled examples is performed separately
                if noise_std > 0:
                    # Corrupted encoder
                    # batch normalization + noise
                    z = join(self.batch_norm_noisy[l-1](z_pre_l), self.batch_norm_noisy[l-1](z_pre_u))
                    z += (torch.randn_like(z) * noise_std)
                else:
                    # Clean encoder
                    # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                    z = join(self.batch_norm_clean_labelled[l-1](z_pre_l),
                             self.batch_norm_clean_unlabelled[l-1](z_pre_u))

            else:
                # Evaluation batch normalization
                z = self.batch_norm_clean_labelled[l-1](z_pre)

            if l == L:
                # use softmax activation in output layer
                # softmax done outside to allow crossentropyloss
                h = self.gamma * (z + self.beta[l-1])
            else:
                # use ReLU activation in hidden layers
                h = F.relu(z + self.beta[l-1])

            d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
        d['labeled']['h'][L], d['unlabeled']['h'][L] = split_lu(h)
        return h, d


class decoders(Module):
    def __init__(self):
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
        for l in range(L, -1, -1):
            z_c = corr['unlabeled']['z'][l]
            if l == L:
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


class Ladder(Module):
    def __init__(self):
        super(Ladder, self).__init__()

        self.encoders = encoders()
        self.decoders = decoders()

    def forward_encoders(self, inputs, noise_std, train):
        return self.encoders.forward(inputs, noise_std, train)

    def forward_decoders(self, y_c, corr, clean):
        return self.decoders.forward(y_c, corr, clean)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('=====Loading Data=====')

unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = load_MNIST_data(100, 49900, True, True)

labelled_loader = DataLoader(supervised_dataset, batch_size=100, shuffle=True)
unlabelled_loader = DataLoader(unsupervised_dataset, batch_size=100, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=validation_dataset.__len__())

ladder = Ladder()

print('=====Encoders=====')
print(ladder.encoders)

print('=====Decoders=====')
print(ladder.decoders)

optimizer = torch.optim.Adam(ladder.parameters(), lr=0.02)
supervised_cost_function = nn.CrossEntropyLoss()
unsupervised_cost_function = nn.MSELoss(reduction="mean")

def evaluate(dataloader):
    ladder.eval()

    correct = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.float()
            labels = labels

            outputs, _ = ladder.forward_encoders(data, 0.0, False)

            _, predicted = torch.max(F.softmax(outputs).data, 1)
            correct += (predicted == labels).sum().item()

    return correct / len(dataloader.dataset)


print('=====Starting Training=====')

for epoch in range(150):
    for batch_idx, (labelled_data, unlabelled_data) in enumerate(zip(cycle(labelled_loader), unlabelled_loader)):
        ladder.train()

        optimizer.zero_grad()

        labelled_images, labels = labelled_data
        labelled_images = labelled_images.float()

        unlabelled_images = unlabelled_data.float()

        inputs = torch.cat((labelled_images, unlabelled_images), 0)

        y_c, corr = ladder.forward_encoders(inputs, noise_std, True)
        y, clean = ladder.forward_encoders(inputs, 0.0, True)

        z_est_bn = ladder.forward_decoders(F.softmax(y_c), corr, clean)

        cost = supervised_cost_function.forward(labeled(y_c), labels)

        zs = clean['unlabeled']['z']

        u_cost = 0
        for l in range(L, -1, -1):
            # print('z_est', z_est_bn[l][0])
            # print('z', zs[l][0])
            # print(unsupervised_cost_function.forward(z_est_bn[l][0], zs[l][0]))
            u_cost += unsupervised_cost_function.forward(z_est_bn[l], zs[l]) * denoising_cost[l]

        loss = cost + u_cost

        loss.backward()
        optimizer.step()

        print('Epoch: {} Supervised Cost: {} Unsupervised Cost: {} Validation Accuracy: {}'.format(
            epoch, cost.item(), u_cost.item(), evaluate(validation_loader)
        ))
