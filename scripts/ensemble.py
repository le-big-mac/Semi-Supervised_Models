import argparse
import pickle
from utils.datautils import *
from sklearn.metrics import matthews_corrcoef
import torch
import math
import torch.nn.functional as F
import statistics

m2_path = './outputs/tcga_minmax_m2/m2/results'
ladder_path = './outputs/tcga_standard/ladder/results'

num_labelled = [100, 500, 1000, 100000]

for n in num_labelled:
    m2_predictions = []
    ladder_predictions = []
    actual = []
    m2_accs = []
    ladder_accs = []

    for i in range(5):
        m2_dict = pickle.load(open('{}/{}_DROP_SAMPLES_{}_classification.p'.format(m2_path, i, n), 'rb'))
        m2_acc = pickle.load(open('{}/{}_DROP_SAMPLES_{}_test_results.p'.format(m2_path, i, n), 'rb'))
        ladder_dict = pickle.load(open('{}/{}_DROP_SAMPLES_{}_classification.p'.format(ladder_path, i, n), 'rb'))
        ladder_acc = pickle.load(open('{}/{}_DROP_SAMPLES_{}_test_results.p'.format(ladder_path, i, n), 'rb'))

        for j in [0, 1]:
            pred_m2, real = next(v for k, v in m2_dict.items() if k.startswith('{}_{}'.format(i, j)))
            pred_ladder, _ = next(v for k, v in ladder_dict.items() if k.startswith('{}_{}'.format(i, j)))
            ma = next(v for k, v in m2_acc.items() if k.startswith('{}_{}'.format(i, j)))
            la = next(v for k, v in ladder_acc.items() if k.startswith('{}_{}'.format(i, j)))

            m2_predictions.append(pred_m2)
            ladder_predictions.append(pred_ladder)
            actual.append(real)
            m2_accs.append(ma)
            ladder_accs.append(la)


    ensemble_pred = [(F.softmax(m) + F.softmax(l))/2 for m, l in zip(m2_predictions, ladder_predictions)]
    ensemble_pred = [torch.max(p.data, 1)[1] for p in ensemble_pred]
    accuracies = [(e == a).sum().item()/len(a) for e, a in zip(ensemble_pred, actual)]

    for i in range(10):
        print(len(m2_predictions[i]))
        print(len(ladder_predictions[i]))

    m2_predictions = torch.cat(m2_predictions)
    ladder_predictions = torch.cat(ladder_predictions)
    actual = torch.cat(actual)

    predictions = (F.softmax(m2_predictions) + F.softmax(ladder_predictions))/2
    _, predictions = torch.max(predictions.data, 1)

    accuracy = (predictions.cpu() == actual).sum().item()/len(actual)
    confidence = 1.96*math.sqrt((accuracy*(1-accuracy))/len(actual))
    mcc = matthews_corrcoef(actual, predictions)

    print('{} labelled accuracy: {} +- {}'.format(n, accuracy, confidence))
    print('mcc: {}'.format(mcc))
    print('accuracies: {}'.format(accuracies))

    for a in [m2_accs, ladder_accs]:
        diff = [a - b for a, b in zip(accuracies, a)]
        print(diff)
        print(statistics.mean(accuracies))
        print(statistics.mean(a))
        mean = statistics.mean(diff)
        stdev = statistics.stdev(diff)

        t = (math.sqrt(len(diff)) * mean) / stdev

        print(t)
