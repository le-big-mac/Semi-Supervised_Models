import torch
from torch import nn
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, input_size, output_size, activation_string, noise_level, device):
        super(Encoder, self).__init__()

        self.W = nn.Linear(input_size, output_size, bias=False)

        # softmax and relu are our only options
        self.activation_string = activation_string

        self.noise = noise_level
        self.device = device

        # Batch Normalization
        # For Relu Beta of batch-norm is redundant, hence only Gamma is trained
        # For Softmax Beta, Gamma are trained
        # batch-normalization bias
        self.batch_norm_noisy = nn.BatchNorm1d(output_size, affine=False)
        self.batch_norm_clean = nn.BatchNorm1d(output_size, affine=False)

        self.beta = nn.Parameter(torch.zeros(1, output_size).to(device))
        if activation_string == 'softmax':
            self.gamma = nn.Parameter(torch.ones(1, output_size).to(device))

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


class StackedEncoders(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation_strings, noise_level, device):
        super(StackedEncoders, self).__init__()

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

        zs.reverse()
        z_pres.reverse()

        return h, zs, z_pres

    def forward_noisy(self, x):
        # h_tilde(0) = z_tilde(0)
        h_tilde = x + self.noise*torch.randn_like(x).to(self.device)
        z_tildes = [h_tilde]
        for encoder in self.encoders:
            h_tilde, z_tilde = encoder.forward_noisy(h_tilde)

            z_tildes.append(z_tilde)

        z_tildes.reverse()

        return h_tilde, z_tildes
