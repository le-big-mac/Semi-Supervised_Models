import argparse
import pickle
from utils.datautils import *
import statistics
import math

parser = argparse.ArgumentParser(description='Take arguments to construct model')
parser.add_argument('model1_path', type=str)
parser.add_argument('model2_path', type=str)
parser.add_argument('--imputation_type', type=str, choices=[i.name.upper() for i in ImputationType],
                    default='DROP_SAMPLES')
args = parser.parse_args()

results1_path = args.model1_path
results2_path = args.model2_path

num_labelled = [100, 500, 1000, 100000]

for n in num_labelled:
    accuracies1 = []
    accuracies2 = []

    for i in range(5):
        accuracy_dict1 = pickle.load(open('{}/{}_{}_{}_test_results.p'.format(results1_path, i, args.imputation_type, n), 'rb'))
        accuracy_dict2 = pickle.load(open('{}/{}_{}_{}_test_results.p'.format(results2_path, i, args.imputation_type, n), 'rb'))

        for j in [0, 1]:
            accuracy1 = next(v for k, v in accuracy_dict1.items() if k.startswith('{}_{}'.format(i, j)))
            accuracy2 = next(v for k, v in accuracy_dict2.items() if k.startswith('{}_{}'.format(i, j)))

            accuracies1.append(accuracy1)
            accuracies2.append(accuracy2)

    diff = [a - b for a, b in zip(accuracies1, accuracies2)]
    print(diff)
    print(statistics.mean(accuracies1))
    print(statistics.mean(accuracies2))
    mean = statistics.mean(diff)
    stdev = statistics.stdev(diff)

    t = (math.sqrt(len(diff)) * mean)/stdev

    print(t)
