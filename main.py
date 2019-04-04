import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
from utils.datautils import load_MNIST_data, save_results
from Models import SimpleNetwork, PretrainingNetwork, SDAE, SimpleM1, M1, M2Runner, LadderNetwork
from Saliency import VanillaSaliency, GuidedSaliency
import random
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_path = './Models/state'

parser = argparse.ArgumentParser(description='Take arguments to construct model')
parser.add_argument('model', type=str,
                    choices=['simple', 'pretraining', 'sdae', 'simple_m1', 'm1', 'm2', 'ladder'],
                    help="Choose which model to run"
                    )
parser.add_argument('--saliency', type=bool, default=False, help="If true compute saliencies for test set")
# parser.add_argument("unsupervised_file", type=str,
#                         help="Relative path to file containing data for unsupervised training")
# parser.add_argument("supervised_data_file", type=str,
#                         help="Relative path to file containing the input data for supervised training")
# parser.add_argument("supervised_labels_file", type=str,
#                         help="Relative path to file containing the output data for supervised training")
# parser.add_argument("--model", dest="unsupervised_model_file", type=str, default="model.pt",
#                     help="Relative path to file to store the trained unsupervised model")
# parser.add_argument("--batch_size", dest="batch_size", type=int, default=500, help="Batch size to use in training")
# parser.add_argument("--epochs_unsupervised", dest="num_epochs_unsupervised", type=int, default=100,
#                     help="Number of epochs to train for unsupervised")
# parser.add_argument("--epochs_supervised", dest="num_epochs_supervised", type=int, default=100,
#                     help="Number of epochs to train for supervised")
# parser.add_argument("--num_folds", dest="num_folds", type=int, default=10,
#                     help="Number of folds for cross validation")
# parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-3,
#                     help="Learning rate for gradient descent")

args = parser.parse_args()

model_name = args.model
unsupervised_batch_size = 1000
batch_size = 100

dataset_name = 'MNIST'
unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
    load_MNIST_data(100, 50000, True, True)

unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=unsupervised_batch_size, shuffle=True)
supervised_dataloader = DataLoader(supervised_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=validation_dataset.__len__())
test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__())

if not os.path.exists(state_path):
    os.mkdir(state_path)
if not os.path.exists('{}/{}'.format(state_path, model_name)):
    os.mkdir('{}/{}'.format(state_path, model_name))



model = None
dataloaders = (supervised_dataloader, unsupervised_dataloader, validation_dataloader)

if model_name == 'simple':
    model = SimpleNetwork(784, [1000, 500, 250, 250, 250], 10, dataset_name, device)
    dataloaders = (supervised_dataloader, validation_dataloader)
elif model_name == 'pretraining':
    model = PretrainingNetwork(784, [1000, 500, 250, 250, 250], 10, lambda x: x, nn.Sigmoid(), dataset_name, device)
elif model_name == 'sdae':
    model = SDAE(784, [1000, 500, 250, 250, 250], 10, nn.ReLU(), dataset_name, device)
elif model_name == 'simple_m1':
    model = SimpleM1(784, [256, 128], 32, [32], 10, lambda x: x, nn.Sigmoid(), dataset_name, device)
elif model_name == 'm1':
    model = M1(784, [256, 128], 32, [32], 10, nn.Sigmoid(), dataset_name, device)
elif model_name == 'm2':
    model = M2Runner(784, [256, 128], [256], 32, 10, nn.Sigmoid(), dataset_name, device)
    un = DataLoader(unsupervised_dataset, batch_size=batch_size, shuffle=True)
    dataloaders = (supervised_dataloader, un, validation_dataloader)
elif model_name == 'ladder':
    model = LadderNetwork(784, [1000, 500, 250, 250, 250], 10, [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10],
                          dataset_name, device)
    un = DataLoader(unsupervised_dataset, batch_size=batch_size, shuffle=True)
    dataloaders = (supervised_dataloader, un, validation_dataloader)

if not args.saliency:
    epochs_list = []
    losses_list = []
    validation_accs_list = []
    results_list = []
    for i in range(5):
        epochs, losses, validation_accs = model.train(*dataloaders)
        results = model.test(test_dataloader)

        epochs_list.append(epochs)
        losses_list.append(losses)
        validation_accs_list.append(validation_accs)
        results_list.append([results])

    save_results(epochs_list, dataset_name, model_name, 'epochs')
    save_results(losses_list, dataset_name, model_name, 'losses')
    save_results(validation_accs_list, dataset_name, model_name, 'validation_accs')
    save_results(results_list, dataset_name, model_name, 'test_accuracy')

else:
    unsupervised_dataset, supervised_dataset, validation_dataset, test_dataset = \
        load_MNIST_data(60000, 20, True, True)
    supervised_dataloader = DataLoader(supervised_dataset, batch_size=1000, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=validation_dataset.__len__())
    dataloaders = (supervised_dataloader, validation_dataloader)

    model.train(*dataloaders)

    input_idx = random.choice(range(len(test_dataset)))
    input = test_dataset[input_idx][0].unsqueeze(0)

    output = model.classify(input)

    _, prediction = output.max(1)

    print(prediction)

    saliency = GuidedSaliency(model.Classifier, device).generate_saliency(input, prediction)

    if device.type == 'cuda':
        saliency = saliency.cpu()

    pos_map = saliency.clamp(min=0)
    pos_map = pos_map / pos_map.max()

    neg_map = - saliency.clamp(max=0)
    neg_map = neg_map / neg_map.max()

    abs_map = saliency.abs()
    abs_map = abs_map / abs_map.max()

    input = input.view(28, 28)
    pos_map = pos_map.view(28, 28)
    neg_map = neg_map.view(28, 28)
    abs_map = abs_map.view(28, 28)

    plt.imsave('original.png', input, cmpa='gray')
    plt.imsave('pos.png', pos_map, cmpa='gray')
    plt.imsave('neg.png', neg_map, cmpa='gray')
    plt.imsave('abs.png', abs_map, cmpa='gray')

    # figure = plt.figure(figsize=(8, 8), facecolor='w')
    #
    # plt.subplot(2, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(input, cmap="gray")
    #
    # plt.subplot(2, 2, 2)
    # plt.title("Positive Saliency")
    # plt.imshow(pos_map, cmap='gray')
    #
    # plt.subplot(2, 2, 3)
    # plt.title("Negative Saliency")
    # plt.imshow(neg_map, cmap='gray')
    #
    # plt.subplot(2, 2, 4)
    # plt.title("Absolute Saliency")
    # plt.imshow(abs_map, cmap='gray')
    #
    # plt.show()


