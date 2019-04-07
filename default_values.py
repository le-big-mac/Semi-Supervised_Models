import torch
from torch import nn
from Models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_path = './Models/state'