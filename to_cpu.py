import argparse
import pickle
from utils.datautils import *
import torch
import math
import os

dataset_names = ['tcga_minmax', 'tcga_minmax_m2', 'tcga_standard']
models = ['simple', 'm1', 'sdae', 'm2', 'ladder']
num_labelled = [100, 500, 1000, 100000]

for dataset_name in dataset_names:
    for model in models:

        if os.path.exists('./outputs/{}/{}'.format(dataset_name, model)):
            results_path = './outputs/{}/{}/results'.format(dataset_name, model)

            prediction_dict_cpu = {}

            for n in num_labelled:
                for i in range(5):
                    prediction_dict = pickle.load(open('{}/{}_DROP_SAMPLES_{}_classification.p'.format(results_path, i, n), 'rb'))

                    prediction_dict_cpu = {k: (p.cpu(), r) for (k, (p, r)) in prediction_dict.items()}

                    pickle.dump(open('{}/{}_DROP_SAMPLES_{}_classification.p'.format(results_path, i, n), 'wb'))
