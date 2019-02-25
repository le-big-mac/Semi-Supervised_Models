import torch
from torch import nn
from itertools import cycle
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import Accuracy


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, activation_string, noise_level, device):
        super(Encoder, self).__init__()

        self.W = nn.Linear(input_size, output_size, bias=False)

        # softmax and relu are our only options
        self.activation_string = activation_string

        self.batch_norm_noisy = nn.BatchNorm1d(output_size, affine=False)
        self.batch_norm_clean = nn.BatchNorm1d(output_size, affine=False)

        self.beta = nn.Parameter(torch.zeros(1, output_size).to(device))
        if activation_string == 'softmax':
            self.gamma = nn.Parameter(torch.ones(1, output_size).to(device))

        self.noise = noise_level
        self.device = device

    def forward_clean(self, x):
        z_pre = self.W(x)

        z = self.batch_norm_clean(z_pre)

        z_beta = z + self.beta.expand_as(z)

        if self.activation_string == 'relu':
            h = F.relu(z_beta)
        else:
            # if this is the output (softmax) layer we do the softmax during training because of PyTorch's weird loss
            # functions
            h = z_beta * self.gamma.expand_as(z_beta)

        # z_pre used in reconstruction cost
        # loss should be back-propagated through z_tilde not z and z_pre
        return h, z.detach().clone(), z_pre.detach().clone()

    def forward_noisy(self, x):
        z_tilde_pre = self.W(x)

        z_tilde = self.batch_norm_noisy(z_tilde_pre) + self.noise*torch.randn_like(z_tilde_pre).to(self.device)

        z_tilde_beta = z_tilde + self.beta.expand_as(z_tilde)

        if self.activation_string == 'relu':
            h_tilde = F.relu(z_tilde_beta)
        else:
            h_tilde = z_tilde_beta * self.gamma.expand_as(z_tilde_beta)

        return h_tilde, z_tilde.clone()

    def forward(self, x, clean):
        if clean:
            return self.forward_clean(x)
        else:
            return self.forward_noisy(x)


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation_strings, noise_level, device):
        super(Classifier, self).__init__()

        dimensions = [input_size] + hidden_dimensions + [num_classes]
        encoders = [Encoder(dimensions[i], dimensions[i+1], activation_strings[i], noise_level, device)
                    for i in list(range(len(dimensions)-1))]

        self.encoders = nn.ModuleList(encoders)

        self.noise = noise_level
        self.device = device

    def forward_clean(self, x):
        zs = [x]
        z_pres = [x]

        h = x
        for encoder in self.encoders:
            h, z, z_pre = encoder.forward_clean(h)

            zs.append(z)
            z_pres.append(z_pre)

        return h, zs, z_pres

    def forward_noisy(self, x):
        # h_tilde(0) = z_tilde(0)
        h_tilde = x + self.noise*torch.randn_like(x).to(self.device)
        z_tildes = [h_tilde]
        for encoder in self.encoders:
            h_tilde, z_tilde = encoder.forward_noisy(h_tilde)

            z_tildes.append(z_tilde)

        return h_tilde, z_tildes

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
        mu_l = self.a1.expand_as(u_l) * torch.sigmoid(self.a2.expand_as(u_l) * u_l + self.a3.expand_as(u_l)) + \
               self.a4.expand_as(u_l) * u_l + self.a5.expand_as(u_l)

        v_l = self.a6.expand_as(u_l) * torch.sigmoid(self.a7.expand_as(u_l) * u_l + self.a8.expand_as(u_l)) + \
               self.a9.expand_as(u_l) * u_l + self.a10.expand_as(u_l)

        z_hat_l = (z_tilde_l - mu_l) * v_l + mu_l

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

        i = 0
        for decoder, z_tilde, z_pre_l in zip(self.decoders, z_tildes, z_pre_layers):
            z_hat_l = decoder.forward(z_tilde, z_hat_l)

            z_hats.append(z_hat_l)

            assert(z_hat_l.size() == z_pre_l.size())

            mu = z_pre_l.mean(dim=0)

            # TODO: data has areas with no vairance (image corners) leading to 0 stddev and problems
            # currently using hack - fix this or determine it's ok
            sigma = z_pre_l.std(dim=0) + 1e-4
            ones = torch.ones(z_pre_l.size()[0], 1).to(self.device)

            # if i==6:
            #     print(mu)
            #     print(sigma)
            #     print(z_hat_l[0])
            #     print((z_hat_l - mu.expand_as(z_hat_l))[0])
            #     print(torch.div(z_hat_l - mu.expand_as(z_hat_l), ones.mm(torch.unsqueeze(sigma, 0))))

            z_hats_BN.append(torch.div(z_hat_l - mu.expand_as(z_hat_l), ones.mm(torch.unsqueeze(sigma, 0))))

            i += 1

        return z_hats, z_hats_BN


class Ladder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation_strings, noise_level, device):
        super(Ladder, self).__init__()

        self.Classifier = Classifier(input_size, hidden_dimensions, num_classes, activation_strings, noise_level,
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

        self.Ladder = Ladder(input_size, hidden_dimensions, num_classes, activation_strings, noise_level, device)
        self.device = device
        self.unsupervised_loss_multipliers = unsupervised_loss_multipliers
        self.unsupervised_criterion = nn.MSELoss()
        self.supervised_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.Ladder.parameters(), lr=1e-3)
        self.state_path = 'Models/state/ladder.pt'
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

            y_tilde_un_pre, z_tildes_un = self.Ladder.encoder_forward_noisy(unlabeled_data)
            y_tilde_lab_pre, z_tildes_lab = self.Ladder.encoder_forward_noisy(labeled_data)

            y_un_pre, zs_un, z_pres_un = self.Ladder.encoder_forward_clean(unlabeled_data)
            y_lab_pre, zs_lab, z_pres_lab = self.Ladder.encoder_forward_clean(labeled_data)

            y_tilde_un = F.softmax(y_tilde_un_pre, dim=1)
            y_tilde_lab = F.softmax(y_tilde_lab_pre, dim=1)

            assert(len(z_tildes_un) == len(z_pres_un))

            # reverse lists because need to be in opposite order for decoders
            z_hats_un, z_hats_BN_un = self.Ladder.decoder_forward(y_tilde_un, z_tildes_un[::-1], z_pres_un[::-1])
            z_hats_lab, z_hats_BN_lab = self.Ladder.decoder_forward(y_tilde_lab, z_tildes_lab[::-1], z_pres_lab[::-1])

            supervised_cost = self.supervised_criterion(y_tilde_lab_pre, labels)
            total_supervised_loss += supervised_cost.item()

            unsupervised_cost = 0.
            for multiplier, z, z_hat_BN in zip(self.unsupervised_loss_multipliers, zs_un[::-1], z_hats_BN_un):
                # print(multiplier)
                # print(self.unsupervised_criterion(z, z_hat_BN))
                unsupervised_cost += multiplier*self.unsupervised_criterion(z, z_hat_BN)
            for multiplier, z, z_hat_BN in zip(self.unsupervised_loss_multipliers, zs_lab[::-1], z_hats_BN_lab):
                # print(multiplier)
                # print(self.unsupervised_criterion(z, z_hat_BN))
                # print(z[0])
                # print(z_hat_BN[0])
                unsupervised_cost += multiplier*self.unsupervised_criterion(z, z_hat_BN)

            total_unsupervised_loss += unsupervised_cost.item()

            cost = supervised_cost + unsupervised_cost

            cost.backward()
            self.optimizer.step()

            if i % 10 == 0:
                validation_acc = self.accuracy(validation_dataloader)
                print('Epoch: {} Supervised Loss: {} Unsupervised Loss {} Validation Accuracy: {}'
                      .format(epoch, supervised_cost.item(), unsupervised_cost.item(), validation_acc))

        return total_supervised_loss/supervised_samples, total_unsupervised_loss/unsupervised_samples

    def accuracy(self, dataloader):
        self.Ladder.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                outputs, _, _ = self.Ladder(data.to(self.device))
                labels = labels.to(self.device)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        return correct / len(dataloader.dataset)

    def reset_model(self):
        self.Ladder.load_state_dict(torch.load(self.state_path))
        self.optimizer = torch.optim.Adam(self.Ladder.parameters(), lr=1e-3)

    def full_train(self, unsupervised_dataset, train_dataset, validation_dataset=None):
        self.reset_model()

        def print_grad_backward(module, grad_input, grad_output):
            print('Inside {} backward'.format(module.__class__.__name__))
            print('grad_input[0]: {}'.format(grad_input[0]))
            print('grad_output[0]: {}'.format(grad_output[0]))

        for decoder in self.Ladder.StackedDecoders.decoders:
            decoder.register_backward_hook(print_grad_backward)
        for encoder in self.Ladder.Classifier.encoders:
            encoder.register_backward_hook(print_grad_backward)

        # TODO: don't use arbitrary values for batch size
        unsupervised_dataloader = DataLoader(dataset=unsupervised_dataset, batch_size=100, shuffle=True)
        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(100):

            self.train_one_epoch(epoch, supervised_dataloader, unsupervised_dataloader, validation_dataloader)
            print('Epoch: {} Validation Acc: {}'.format(epoch, self.accuracy(validation_dataloader)))

    def full_test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return self.accuracy(test_dataloader)
