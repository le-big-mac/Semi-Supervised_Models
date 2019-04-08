import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
from utils.datautils import load_MNIST_data, load_toy_data, save_results
from Models import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_path = './Models/state'


def get_datasets(args):
    dataset_name = args.dataset
    num_labelled = args.num_labelled
    num_unlabelled = args.num_unlabelled

    datasets = None
    if dataset_name == 'mnist':
        datasets = load_MNIST_data(num_labelled, num_unlabelled, True, True)
        input_size = 784
        output_size = 10
    elif dataset_name == 'toy':
        datasets, input_size, output_size = load_toy_data(num_labelled, num_unlabelled, True, True)

    return datasets, input_size, output_size


def get_models_and_dataloaders(args, datasets, input_size, output_size):
    model = None
    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = datasets
    batch_size = 100
    pretraining_batch_size = 1000

    unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=pretraining_batch_size, shuffle=True)
    supervised_dataloader = DataLoader(supervised_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=validation_dataset.__len__())
    test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__())

    # default
    train_dataloaders = (unsupervised_dataloader, supervised_dataloader, validation_dataloader)

    model_name = args.model
    dataset_name = args.dataset
    classifier_layers = args.classifier_layers

    if model_name == 'simple':
        model = SimpleNetwork(input_size, classifier_layers, output_size, dataset_name, device)
        train_dataloaders = (supervised_dataloader, validation_dataloader)

    elif model_name == 'pretraining':
        model = PretrainingNetwork(input_size, classifier_layers, output_size, nn.Sigmoid(), dataset_name, device)

    elif model_name == 'sdae':
        model = SDAE(input_size, classifier_layers, output_size, dataset_name, device)

    elif model_name == 'simple_m1':
        model = SimpleM1(input_size, args.autoencoder_layers, args.latent_size, classifier_layers, output_size,
                         lambda x: x, dataset_name, device)

    elif model_name == 'm1':
        model = M1(input_size, args.autoencoder_layers, args.latent_size, classifier_layers, output_size,
                   lambda x: x, dataset_name, device)

    elif model_name == 'm2':
        model = M2Runner(input_size, args.autoencoder_layers, classifier_layers, args.latent_size, output_size,
                         lambda x: x, dataset_name, device)

        unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=batch_size, shuffle=True)
        train_dataloaders = (unsupervised_dataloader, supervised_dataloader, validation_dataloader)

    elif model_name == 'ladder':
        model = LadderNetwork(input_size, classifier_layers, output_size,
                              [1000.0, 10.0] + [0.1] * (len(classifier_layers)), dataset_name, device)

        unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=batch_size, shuffle=True)
        train_dataloaders = (unsupervised_dataloader, supervised_dataloader, validation_dataloader)

    return model, train_dataloaders, test_dataloader


def main():
    parser = argparse.ArgumentParser(description='Take arguments to construct model')
    parser.add_argument('model', type=str,
                        choices=['simple', 'pretraining', 'sdae', 'simple_m1', 'm1', 'm2', 'ladder'],
                        help="Choose which model to run"
                        )
    parser.add_argument('dataset', type=str, choices=['mnist', 'toy', 'tcga', 'metabric'], help='Dataset to run on')
    parser.add_argument('num_labelled', type=int)
    parser.add_argument('num_unlabelled', type=int)
    parser.add_argument('classifier_layers', type=int, nargs='+', help='Hidden layer sizes')
    parser.add_argument('--latent_size', type=int)
    parser.add_argument('--autoencoder_layers', type=int)
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs to run for')
    parser.add_argument('--comparison', type=bool, default=False,
                        help='Saves data on validation and losses per iteration for comparison between models (will '
                             'slow down training)')
    args = parser.parse_args()

    model_name = args.model

    if not os.path.exists(state_path):
        os.mkdir(state_path)
    if not os.path.exists('{}/{}'.format(state_path, model_name)):
        os.mkdir('{}/{}'.format(state_path, model_name))

    dataset_name = args.dataset
    datasets, input_size, output_size = get_datasets(args)

    epochs_list = []
    losses_list = []
    validation_accs_list = []
    results_list = []
    for i in range(5):
        model, train_dataloaders, test_dataloader = \
            get_models_and_dataloaders(args, datasets, input_size, output_size)

        epochs, losses, validation_accs = model.train_model(args.max_epochs, train_dataloaders)
        results = model.test_model(test_dataloader)

        epochs_list.append(epochs)
        losses_list.append(losses)
        validation_accs_list.append(validation_accs)
        results_list.append([results])

    save_results(results_list, dataset_name, model_name, 'test_accuracy')
    if args.comparison:
        save_results(epochs_list, dataset_name, model_name, 'epochs')
        save_results(losses_list, dataset_name, model_name, 'losses')
        save_results(validation_accs_list, dataset_name, model_name, 'validation_accs')


if __name__ == '__main__':
    main()