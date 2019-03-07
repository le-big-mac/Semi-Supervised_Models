import numpy as np
import torch
from torch import nn


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

        for decoder, z_tilde, z_pre_l in zip(self.decoders, z_tildes, z_pre_layers):
            z_hat_l = decoder.forward(z_tilde, z_hat_l)

            z_hats.append(z_hat_l)

            assert(z_hat_l.size() == z_pre_l.size())

            # ones = torch.ones(z_pre_l.size()[0], 1).to(self.device)
            #
            # mean = torch.mean(z_pre_l, 0)
            #
            # noise_var = np.random.normal(loc=0.0, scale=1 - 1e-10, size=z_pre_l.size())
            # if self.device == 'cuda':
            #     var = np.var(z_pre_l.data.cpu().numpy() + noise_var, axis=0).reshape(1, z_pre_l.size()[1])
            # else:
            #     var = np.var(z_pre_l.data.numpy() + noise_var, axis=0).reshape(1, z_pre_l.size()[1])
            # var = torch.FloatTensor(var)
            #
            # if self.device == 'cuda':
            #     z_hat_l = z_hat_l.cpu()
            #     ones = ones.cpu()
            #     mean = mean.cpu()
            # hat_z_normalized = torch.div(z_hat_l - ones.mm(torch.unsqueeze(mean, 0)), ones.mm(torch.sqrt(var + 1e-10)))
            #
            # if self.device == 'cuda':
            #     hat_z_normalized = hat_z_normalized.cuda()
            #
            # z_hats_BN.append(hat_z_normalized)

            # if decoder == self.decoders[-1]:
            if True:
                hat_z_normalized = z_hat_l
            else:
                mean = z_pre_l.mean(dim=0)
                var = z_pre_l.var(dim=0)

                hat_z_normalized = (z_hat_l - mean.expand_as(z_hat_l)) / torch.sqrt(var + 1e-10)

            z_hats_BN.append(hat_z_normalized)

        return z_hats, z_hats_BN
