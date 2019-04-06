import torch
from torch import nn
from Models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_path = './Models/state'

mnist_models = {
    'simple': SimpleNetwork(784, [1000, 500, 250, 250, 250], 10, 'mnist', device),
    'pretraining': PretrainingNetwork(784, [1000, 500, 250, 250, 250], 10, nn.Sigmoid(), 'mnist', device),
    'sdae': SDAE(784, [1000, 500, 250, 250, 250], 10, nn.ReLU(), 'mnist', device),
    'simple_m1': SimpleM1(784, [256, 128], 32, [32], 10, nn.Sigmoid(), 'mnist', device),
    'm1': M1(784, [256, 128], 32, [32], 10, nn.Sigmoid(), 'mnist', device),
    'm2': M2Runner(784, [256, 128], [256], 32, 10, nn.Sigmoid(), 'mnist', device),
    'ladder': LadderNetwork(784, [1000, 500, 250, 250, 250], 10, [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10], 'mnist',
                            device)
}