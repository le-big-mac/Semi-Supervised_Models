import torch
import os
import csv
import sys
from torch import nn
from itertools import cycle
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import Arguments, Datasets, KFoldSplits, LoadData


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, activation, noise_level, is_linear, device):
        super(Encoder, self).__init__()

        self.W = nn.Linear(input_size, output_size, bias=False)
        self.activation = activation
        self.batch_norm_noisy = nn.BatchNorm1d(output_size, affine=False)
        self.batch_norm_clean = nn.BatchNorm1d(output_size, affine=False)
        self.beta = nn.Parameter(torch.zeros(1, output_size).to(device))
        self.is_linear = is_linear

        if not is_linear:
            self.gamma = nn.Parameter(torch.ones(1, output_size).to(device))

        self.noise = noise_level
        self.device = device

    def forward_clean(self, x):
        z_pre = self.W(x)
        z = self.batch_norm_clean(z_pre)

        z_beta = z + self.beta.expand_as(z)

        if self.is_linear:
            h = self.activation(z_beta)
        else:
            ones = torch.ones(z.size()[0], 1).to(self.device)
            gamma = ones.mm(self.gamma)
            h = self.activation(torch.mul(gamma, z_beta))

        # z_pre used in reconstruction cost
        # loss should be back-propagated through z_tilde not z and z_pre
        return h, z.detach().clone(), z_pre.detach().clone()

    def forward_noisy(self, x):
        temp = self.W(x)
        z_pre_tilde = temp + self.noise*torch.randn_like(temp).to(self.device)
        z_tilde = self.batch_norm_noisy(z_pre_tilde)
        h_tilde = self.activation(z_tilde + self.beta.expand_as(z_tilde))

        return h_tilde, z_tilde

    def forward(self, x, clean):
        if clean:
            return self.forward_clean(x)
        else:
            return self.forward_noisy(x)


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activations, is_linear, noise_level, device):
        super(Classifier, self).__init__()

        dimensions = [input_size] + hidden_dimensions + [num_classes]
        encoders = [Encoder(dimensions[i], dimensions[i+1], activations[i], noise_level, is_linear[i], device)
                         for i in list(range(len(dimensions)-1))]

        self.encoders = nn.ModuleList(encoders)

        self.noise = noise_level
        self.device = device

    def forward_clean(self, x):
        zs = [x]

        h = x
        z_pres = [h]
        for encoder in self.encoders:
            h, z, z_pre = encoder.forward_clean(h)

            zs.append(z)
            z_pres.append(z_pre)

        y = h

        return y, zs, z_pres

    def get_zs(self):
        return self.zs, self.z_pres

    def forward_noisy(self, x):
        # h_tilde(0) = z_tilde(0)
        h_tilde = x + self.noise*torch.randn_like(x).to(self.device)
        z_tildes = [h_tilde]
        for encoder in self.encoders:
            h_tilde, z_tilde = encoder.forward_noisy(h_tilde)

            z_tildes.append(z_tilde)

        # unnecessary assignment, keeps notation more in line with paper
        y_tilde = h_tilde

        return y_tilde, z_tildes

    def forward(self, x, clean):
        if clean:
            return self.forward_clean(x)
        else:
            return self.forward_noisy(x)


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, is_bottom, device):
        super(Decoder, self).__init__()

        if is_bottom:
            self.V = lambda x: x
        else:
            self.V = nn.Linear(input_size, output_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(output_size, affine=False)
        self.device = device

        self.a1 = nn.Parameter(torch.zeros(1, output_size).to(device))
        self.a2 = nn.Parameter(torch.ones(1, output_size).to(device))
        self.a3 = nn.Parameter(torch.zeros(1, output_size).to(device))
        self.a4 = nn.Parameter(torch.zeros(1, output_size).to(device))
        self.a5 = nn.Parameter(torch.zeros(1, output_size).to(device))
        self.a6 = nn.Parameter(torch.zeros(1, output_size).to(device))
        self.a7 = nn.Parameter(torch.ones(1, output_size).to(device))
        self.a8 = nn.Parameter(torch.zeros(1, output_size).to(device))
        self.a9 = nn.Parameter(torch.zeros(1, output_size).to(device))
        self.a10 = nn.Parameter(torch.zeros(1, output_size).to(device))

    def g(self, u_l, z_tilde_l):

        ones = torch.ones(z_tilde_l.size()[0], 1).to(self.device)

        m_a1 = ones.mm(self.a1)
        m_a2 = ones.mm(self.a2)
        m_a3 = ones.mm(self.a3)
        m_a4 = ones.mm(self.a4)
        m_a5 = ones.mm(self.a5)
        m_a6 = ones.mm(self.a6)
        m_a7 = ones.mm(self.a7)
        m_a8 = ones.mm(self.a8)
        m_a9 = ones.mm(self.a9)
        m_a10 = ones.mm(self.a10)

        mu_l = torch.mul(m_a1, torch.sigmoid(torch.mul(m_a2, u_l) + m_a3)) + torch.mul(m_a4, u_l) + m_a5

        v_l = torch.mul(m_a6, torch.sigmoid(torch.mul(m_a7, u_l) + m_a8)) + torch.mul(m_a9, u_l) + m_a10

        z_hat_l = torch.mul(z_tilde_l - mu_l, v_l) + mu_l

        return z_hat_l

    def forward(self, z_tilde_l, z_hat_l_plus_1):
        # maybe take in u_l instead, would maybe make stacked decoder easier
        u_l = self.batch_norm(self.V(z_hat_l_plus_1))

        z_hat_l = self.g(u_l, z_tilde_l)

        return z_hat_l


class StackedDecoders(nn.Module):
    def __init__(self, num_classes, hidden_dimensions, input_size, device):
        super(StackedDecoders, self).__init__()

        dimensions = [num_classes] + hidden_dimensions + [input_size]

        decoders = [Decoder(num_classes, num_classes, True, device)]
        decoders.extend([Decoder(dimensions[i], dimensions[i+1], False, device)
                         for i in list(range(len(dimensions)-1))])

        self.decoders = nn.ModuleList(decoders)

        self.device = device

    def forward(self, u_L, z_tildes, z_pre_layers):
        z_hats = []
        z_hats_BN = []
        z_hat_l = u_L

        for decoder, z_tilde, z_pre_l in zip(self.decoders, z_tildes, z_pre_layers):
            z_hat_l = decoder.forward(z_tilde, z_hat_l)

            z_hats.append(z_hat_l)

            assert(z_hat_l.size() == z_pre_l.size())

            mu = z_pre_l.mean(dim=0)
            sigma = z_pre_l.std(dim=0)
            ones = torch.ones(z_pre_l.size()[0], 1).to(self.device)

            z_hats_BN.append(torch.div(z_hat_l - mu.expand_as(z_hat_l), ones.mm(torch.unsqueeze(sigma, 0))))

        return z_hats, z_hats_BN


class Ladder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activations, is_linear, noise_level, device):
        super(Ladder, self).__init__()

        self.Classifier = Classifier(input_size, hidden_dimensions, num_classes, activations, is_linear, noise_level,
                                     device)

        self.StackedDecoders = StackedDecoders(num_classes, hidden_dimensions[::-1], input_size, device)

        self.device = device

    def encoder_forward_clean(self, x):
        return self.Classifier.forward_clean(x)

    def encoder_forward_noisy(self, x):
        return self.Classifier.forward_noisy(x)

    def encoder_get_zs(self):
        return self.Classifier.get_zs()

    def decoder_forward(self, h_L, z_tildes, z_pre_layers):
        return self.StackedDecoders.forward(h_L, z_tildes, z_pre_layers)

    def forward(self, x):
        return self.encoder_forward_clean(x)


class LadderNetwork:
    def __init__(self, input_size, hidden_dimensions, num_classes, activation_strings, noise_level,
                 unsupervised_loss_multipliers, device):

        activations = []
        is_linear = []
        for act in activation_strings:
            if act == 'relu':
                activations.append(nn.ReLU())
                is_linear.append(True)
            elif act == 'softmax':
                activations.append(nn.LogSoftmax(dim=1))
                is_linear.append(False)
            else:
                raise ValueError("Not an available activation function")

        self.Ladder = Ladder(input_size, hidden_dimensions, num_classes, activations, is_linear, noise_level, device)
        self.device = device
        self.unsupervised_loss_multipliers = unsupervised_loss_multipliers
        self.unsupervised_criterion = nn.MSELoss()
        self.supervised_criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.Ladder.parameters(), lr=1e-3)
        self.state_path = 'state/ladder.pt'
        torch.save(self.Ladder.state_dict(), self.state_path)

    def train_one_epoch(self, epoch, supervised_dataloader, unsupervised_dataloader, validation_dataloader):
        total_supervised_loss = 0
        total_unsupervised_loss = 0

        supervised_samples = 0
        unsupervised_samples = 0

        self.Ladder.train()

        # Repeat the supervised dataloader (several full supervised trains for one unsupervised train)
        for i, (supervised, unlabeled_data) in enumerate(zip(cycle(supervised_dataloader), unsupervised_dataloader)):
            self.optimizer.zero_grad()

            labeled_data, labels = supervised
            unlabeled_data = unlabeled_data.float()

            labeled_data.to(self.device)
            labels.to(self.device)
            unlabeled_data.to(self.device)

            supervised_samples += len(labeled_data)
            unsupervised_samples += len(unlabeled_data)

            y_tilde_un, z_tildes_un = self.Ladder.encoder_forward_noisy(unlabeled_data)
            y_tilde_lab, z_tildes_lab = self.Ladder.encoder_forward_noisy(labeled_data)

            y_un, zs_un, z_pres_un = self.Ladder.encoder_forward_clean(unlabeled_data)
            y_lab, zs_lab, z_pres_lab = self.Ladder.encoder_forward_clean(labeled_data)

            assert(len(z_tildes_un) == len(z_pres_un))

            # reverse lists because need to be in opposite order for decoders
            z_hats_un, z_hats_BN_un = self.Ladder.decoder_forward(y_tilde_un, z_tildes_un[::-1], z_pres_un[::-1])
            z_hats_lab, z_hats_BN_lab = self.Ladder.decoder_forward(y_tilde_lab, z_tildes_lab[::-1], z_pres_lab[::-1])

            supervised_cost = self.supervised_criterion(y_tilde_lab, labels)
            total_supervised_loss += supervised_cost.item()

            unsupervised_cost = 0
            for multiplier, z, z_hat_BN in zip(self.unsupervised_loss_multipliers, zs_un[::-1], z_hats_BN_un):
                unsupervised_cost += multiplier*self.unsupervised_criterion(z, z_hat_BN)
            for multiplier, z, z_hat_BN in zip(self.unsupervised_loss_multipliers, zs_lab[::-1], z_hats_BN_lab):
                unsupervised_cost += multiplier*self.unsupervised_criterion(z, z_hat_BN)

            total_unsupervised_loss += unsupervised_cost.item()

            cost = supervised_cost + unsupervised_cost

            cost.backward()
            self.optimizer.step()

            if i % 10 == 0:
                validation_acc = self.validation(epoch, validation_dataloader)
                print('Epoch: {} Supervised Loss: {} Unsupervised Loss {} Validation Accuracy: {}'
                      .format(epoch, total_supervised_loss, total_unsupervised_loss, validation_acc))

        return total_supervised_loss/supervised_samples, total_unsupervised_loss/unsupervised_samples

    def validation(self, epoch, dataloader):
        self.Ladder.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                y, _, _ = self.Ladder(data)
                _, predicted = torch.max(y.data, 1)
                correct += (predicted == labels).sum().item()

        return correct / len(dataloader.dataset)

    def test(self, dataloader):
        self.Ladder.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                y, _, _ = self.Ladder(data)
                _, predicted = torch.max(y.data, 1)
                correct += (predicted == labels).sum().item()

        return correct/len(dataloader.dataset)

    def reset_model(self):
        self.Ladder.load_state_dict(torch.load(self.state_path))
        self.optimizer = torch.optim.Adam(self.Ladder.parameters(), lr=1e-3)

    def full_train(self, unsupervised_dataset, train_dataset, validation_dataset=None):
        self.reset_model()

        # TODO: don't use arbitrary values for batch size
        unsupervised_dataloader = DataLoader(dataset=unsupervised_dataset, batch_size=100, shuffle=True)
        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(20):

            self.train_one_epoch(epoch, supervised_dataloader, unsupervised_dataloader, validation_dataloader)
            # self.validation(epoch, validation_dataloader)

    def full_test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return self.test(test_dataloader)


if __name__ == '__main__':
    mnist_train = datasets.MNIST(root='../data/MNIST', train=True, download=True, transform=None)
    mnist_test = datasets.MNIST(root='../data/MNIST', train=False, download=True, transform=transforms.ToTensor())

    unsupervised_dataset = Datasets.MNISTUnsupervised(mnist_train.train_data[:49000])

    supervised_dataset = Datasets.MNISTSupervised(mnist_train.train_data[49000:50000],
                                                  mnist_train.train_labels[49000:50000])

    validation_dataset = Datasets.MNISTSupervised(mnist_train.train_data[50000:], mnist_train.train_labels[50000:])

    test_dataset = mnist_test

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    ladder = LadderNetwork(784, [1000, 500, 250, 250, 250], 10, ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax'],
                           0.2, [1000, 10, 0.1, 0.1, 0.1, 0.1, 0.1], device)

    print(ladder.Ladder)

    ladder.full_train(unsupervised_dataset, supervised_dataset, validation_dataset)

    # args = Arguments.parse_args()
    #
    # unsupervised_data, supervised_data, supervised_labels = LoadData.load_data(
    #     args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # ladder = LadderNetwork(500, [200], 10, ['relu', 'softmax'], 0.2, [], device)
    #
    # unsupervised_dataset = Datasets.UnsupervisedDataset(unsupervised_data)
    # for test_idx, train_idx in KFoldSplits.k_fold_splits(len(supervised_data), 10):
    #
    #     train_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
    #                                                              [supervised_labels[i] for i in train_idx])
    #     test_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
    #                                                             [supervised_labels[i] for i in test_idx])
    #
    #     ladder.full_train(unsupervised_dataset, train_dataset)
    #
    #     correct_percentage = ladder.full_test(test_dataset)
    #
    #     test_results.append(correct_percentage)
    #
    # if not os.path.exists('../results'):
    #     os.mkdir('../results')
    #     os.mkdir('../results/ladder')
    # elif not os.path.exists('../results/ladder'):
    #     os.mkdir('../results/ladder')
    #
    # accuracy_file = open('../results/ladder/accuracy.csv', 'w')
    # accuracy_writer = csv.writer(accuracy_file)
    #
    # accuracy_writer.writerow(test_results)
