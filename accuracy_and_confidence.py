import argparse
import pickle
from utils.datautils import *
import torch
import math

parser = argparse.ArgumentParser(description='Take arguments to construct model')
parser.add_argument('model', type=str, choices=['simple', 'm1', 'sdae', 'm2', 'ladder'],
                    help="Choose which model to run")
parser.add_argument('num_labelled', type=int, help='Number of labelled examples to use')
parser.add_argument('dataset_name', type=str, help='Folder name output file')
parser.add_argument('--imputation_type', type=str, choices=[i.name.upper() for i in ImputationType],
                    default='DROP_SAMPLES')
args = parser.parse_args()

results_path = './outputs/{}/{}/results'.format(args.dataset_name, args.model)

predictions = []
actual = []

for i in range(5):
    prediction_dict = pickle.load(open('{}/{}_{}_{}_classification.p'.format(results_path, i, args.imputation_type, args.num_labelled), 'rb'))
    for pred, real in prediction_dict.values():
        predictions.append(pred)
        actual.append(real)

        _, p = torch.max(pred.data, 1)
        print('Correct: ' + (p.cpu() == real).sum())
        print('Total: ' + len(real))

predictions = torch.cat(predictions)
actual = torch.cat(actual)

_, predictions = torch.max(predictions.data, 1)

accuracy = (predictions.cpu() == actual).sum().item()/len(actual)
confidence = 1.96*math.sqrt((accuracy*(1-accuracy))/len(actual))

print('Accuracy: {} +- {}'.format(accuracy, confidence))