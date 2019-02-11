import torch
import csv
import shutil
import os
import sys
from torch import nn
from torch import functional as F
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from random import shuffle


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, activation, noise_level, device):
        super(Encoder, self).__init__()

        self.W = nn.Linear(input_size, output_size, bias=False)
        self.activation = activation
        self.batch_norm = nn.BatchNorm1d(output_size, affine=False)
        self.beta = nn.Parameter(torch.zeros(1, output_size)).to(device)
        self.noise = noise_level
        # assume all linear for the moments so don't need gamma

    def forward_clean(self, x):
        z_pre = self.W(x)
        z = self.batch_norm(z_pre)
        h = self.activation(z + self.beta.expand_as(z))

        # z_pre used in reconstruction cost
        return h, z, z_pre

    def forward_noisy(self, x):
        z_pre_tilde = self.W(x) + self.noise*torch.randn(x.size()).to(self.device)
        z_tilde = self.batch_norm(z_pre_tilde)
        h_tilde = self.activation(z_tilde + self.beta.expand_as(z_tilde))

        return h_tilde, z_tilde

    def forward(self, x, clean):
        if clean:
            return self.forward_clean(x)
        else:
            return self.forward_noisy(x)


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activations, noise_level, device):
        super(Classifier, self).__init__()

        dimensions = [input_size] + hidden_dimensions + [num_classes]

        self.encoders = [Encoder(dimensions[i], dimensions[i+1], activations[i], noise_level, device)
                         for i, _ in list(range(len(dimensions)-1))]

        self.noise = noise_level
        self.device = device

    def forward_clean(self, x):
        zs = []
        z_pres = []

        h = x
        for encoder in self.encoders:
            h, z, z_pre = encoder.forward_clean(h)

            zs.append(z)
            z_pres.append(z_pre)

        y = h

        return y, zs, z_pres

    def forward_noisy(self, x):
        z_tildes = []

        # no need to add noise here, added by encoder
        h_tilde = x
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
    def __init__(self, input_size, output_size, device):
        super(Decoder, self).__init__()

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
