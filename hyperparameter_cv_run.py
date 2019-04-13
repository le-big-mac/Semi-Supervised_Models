import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
from utils.datautils import *
from Models import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_path = './Models/state'
results_path = './results'


def get_datasets(args):
    dataset_name = args.dataset
    num_labelled = args.num_labelled
    num_unlabelled = args.num_unlabelled

    if dataset_name == 'mnist':
        datasets = load_MNIST_data(num_labelled, num_unlabelled, True, True)
        input_size = 784
        output_size = 10
    elif dataset_name == 'tcga':
        datasets, input_size, output_size = load_tcga_data(num_labelled, num_unlabelled)

    return datasets, input_size, output_size


hyperparameter_fns = {
    'simple': simple_hyperparameter_loop,
    'ladder': ladder_hyperparameter_loop
}


def main():
    parser = argparse.ArgumentParser(description='Take arguments to construct model')
    parser.add_argument('model', type=str,
                        choices=['simple', 'pretraining', 'sdae', 'simple_m1', 'm1', 'm2', 'ladder'],
                        help="Choose which model to run"
                        )
    parser.add_argument('dataset', type=str, choices=['mnist', 'tcga', 'metabric'], help='Dataset to run on')
    parser.add_argument('num_labelled', type=int)

    args = parser.parse_args()

    model_name = args.model

    if not os.path.exists(state_path):
        os.mkdir(state_path)
    if not os.path.exists('{}/{}'.format(state_path, model_name)):
        os.mkdir('{}/{}'.format(state_path, model_name))
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    if not os.path.exists('{}/{}'.format(results_path, model_name)):
        os.mkdir('{}/{}'.format(results_path, model_name))

    dataset_name = args.dataset
    datasets, input_size, output_size = get_datasets(args)

    if dataset_name == 'mnist':
        u_d, s_d, v_d, t_d = load_MNIST_data(args.num_labelled, 50000, True, True)
        hyperparameter_optimizer = hyperparameter_fns[model_name]

        u_dl = DataLoader(u_d, batch_size=100, shuffle=True)
        s_dl = DataLoader(s_d, batch_size=100, shuffle=True)
        v_dl = DataLoader(v_d, batch_size=v_d.__len__())
        t_dl = DataLoader(t_d, batch_size=t_d.__len__())

        for i in range(5):
            hyperparameter_optimizer(dataset_name, (u_dl, s_dl, v_dl, t_dl), 784, 10, device)

    elif dataset_name == 'tcga':
        dataset = load_tcga_data()


if __name__ == '__main__':
    main()