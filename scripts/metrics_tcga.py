import argparse
import pickle
from utils.datautils import *
import torch
import math
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

parser = argparse.ArgumentParser(description='Compute metrics for evaluating TCGA classification')
parser.add_argument('model', type=str, choices=['simple', 'm1', 'sdae', 'm2', 'ladder'],
                    help="Choose which model to run")
parser.add_argument('num_labelled', type=int, help='Number of labelled examples to use')
parser.add_argument('dataset_name', type=str, help='Folder name output file')
parser.add_argument('--imputation_type', type=str, choices=[i.name.upper() for i in ImputationType],
                    default='DROP_SAMPLES')
args = parser.parse_args()

results_path = '../outputs/{}/{}/results'.format(args.dataset_name, args.model)

predictions = []
actual = []

for i in range(5):
    prediction_dict = pickle.load(open('{}/{}_{}_{}_classification.p'.format(results_path, i, args.imputation_type, args.num_labelled), 'rb'))
    for pred, real in prediction_dict.values():
        predictions.append(pred)
        actual.append(real)

        _, p = torch.max(pred.data, 1)
        print('Correct: {}'.format((p.cpu() == real).sum()))
        print('Total: {}'.format(len(real)))

predictions = torch.cat(predictions)
actual = torch.cat(actual)

_, predictions = torch.max(predictions.data, 1)

accuracy = (predictions == actual).sum().item()/len(actual)
confidence = 1.96*math.sqrt((accuracy*(1-accuracy))/len(actual))
mcc = matthews_corrcoef(actual, predictions)

print('Accuracy: {} +- {}'.format(accuracy, confidence))
print('MCC: {}'.format(mcc))

mat = confusion_matrix(actual, predictions)
labels = pickle.load(open('./data/tcga/labels_drop_samples.p', 'rb'))

df_cm = pd.DataFrame(mat, labels, labels)
figure = plt.figure(figsize=(20, 15))
sn.heatmap(df_cm, annot=True, fmt='g', cbar=False)

figure.savefig('./outputs/confusion/{}_{}_{}.pdf'.format(args.model, args.dataset_name, args.num_labelled),
               bbox_inches='tight', pad_inches=0.1)
