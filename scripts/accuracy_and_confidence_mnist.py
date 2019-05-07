import argparse
import pickle
from utils.datautils import *
import statistics
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

accuracies = []

for i in range(5):
    fold_accuracies = pickle.load(open('{}/{}_{}_test_results.p'.format(results_path, i, args.num_labelled), 'rb'))
    accuracies.extend(list(fold_accuracies.values()))

av_acc = round(statistics.mean(accuracies), 4)
confidence = 1.96*math.sqrt((av_acc*(1-av_acc))/10000)

print('Accuracy: {} +- {}'.format(av_acc, confidence))