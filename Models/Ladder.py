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
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(1366, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)


