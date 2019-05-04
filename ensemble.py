import argparse
import pickle
from utils.datautils import *
import torch
import math
import torch.nn.functional as F

m2_path = '../outputs/tcga_minmax_m2/m2/results'
ladder_path = '../outputs/tcga_standard/ladder/results'

m2_predictions = []
ladder_predictions = []
actual = []

num_labelled = [100, 500, 1000, 100000]

for n in num_labelled:
    for i in range(5):
        m2_dict = pickle.load(open('{}/{}_DROP_SAMPLES_{}_classification.p'.format(m2_path, i, n), 'rb'))
        for pred, real in m2_dict.values():
            m2_predictions.append(pred)
            actual.append(real)
        ladder_dict = pickle.load(open('{}/{}_DROP_SAMPLES_{}_classification.p'.format(ladder_path, i, n), 'rb'))
        for pred, _ in ladder_dict.values():
            ladder_predictions.append(pred)

    m2_predictions = torch.cat(m2_predictions)
    ladder_predictions = torch.cat(ladder_predictions)
    actual = torch.cat(actual)

    predictions = (F.softmax(m2_predictions) + F.softmax(ladder_predictions))/2
    _, predictions = torch.max(predictions.data, 1)

    accuracy = (predictions.cpu() == actual).sum().item()/len(actual)
    confidence = 1.96*math.sqrt((accuracy*(1-accuracy))/len(actual))

    print('{} labelled accuracy: {} +- {}'.format(n, accuracy, confidence))
