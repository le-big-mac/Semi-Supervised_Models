import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
from utils.datautils import load_MNIST_data, save_results
from Models import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_path = './Models/state'


def get_models_and_dataloaders(model_name, dataset_name, datasets, batch_size, unsupervised_batch_size):
    model = None
    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = datasets

    unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=unsupervised_batch_size, shuffle=True)
    supervised_dataloader = DataLoader(supervised_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=validation_dataset.__len__())
    test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__())

    train_dataloaders = (unsupervised_dataloader, supervised_dataloader, validation_dataloader)

    if model_name == 'simple':
        model = SimpleNetwork(784, [397], 10, 1e-3, dataset_name, device)

    elif model_name == 'sdae':
        model = SDAE(784, [1000, 500, 250, 250, 250], 10, 1e-3, dataset_name, device)

    elif model_name == 'm1':
        model = M1(784, [256, 128], 32, [32], 10, lambda x: x, dataset_name, device)

    elif model_name == 'm2':
        model = M2Runner(784, [256, 128], [256], 32, 10, lambda x: x, 1e-3, dataset_name, device)

        unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=batch_size, shuffle=True)
        train_dataloaders = (unsupervised_dataloader, supervised_dataloader, validation_dataloader)

    elif model_name == 'ladder':
        model = LadderNetwork(784, [397], 10, [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10], 1e-3,
                              dataset_name, device)

        unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=batch_size, shuffle=True)
        train_dataloaders = (unsupervised_dataloader, supervised_dataloader, validation_dataloader)

    return model, train_dataloaders, test_dataloader


def main():
    parser = argparse.ArgumentParser(description='Take arguments to construct model')
    parser.add_argument('model', type=str,
                        choices=['simple', 'sdae', 'm1', 'm2', 'ladder'],
                        help="Choose which model to run"
                        )
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs to run for')
    parser.add_argument('--comparison', type=bool, default=False,
                        help='Saves data on validation and losses per iteration for comparison between models (will '
                             'slow down training)')
    args = parser.parse_args()

    model_name = args.model
    unsupervised_batch_size = 1000
    batch_size = 100

    dataset_name = 'MNIST'
    datasets = load_MNIST_data(100, 50000, True, True)

    if not os.path.exists(state_path):
        os.mkdir(state_path)
    if not os.path.exists('{}/{}'.format(state_path, model_name)):
        os.mkdir('{}/{}'.format(state_path, model_name))

    epochs_list = []
    losses_list = []
    validation_accs_list = []
    results_list = []
    for i in range(5):
        model, train_dataloaders, test_dataloader = \
            get_models_and_dataloaders(model_name, dataset_name, datasets, batch_size, unsupervised_batch_size)

        epochs, losses, validation_accs = model.train_model(args.max_epochs, train_dataloaders, True)
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
