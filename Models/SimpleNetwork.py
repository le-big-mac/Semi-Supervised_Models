import torch
import csv
import shutil
import os
import sys
from torch import nn
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from random import shuffle


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()

        self.regression = nn.Sequential(
            nn.Linear(1366, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.regression(x)

parser = argparse.ArgumentParser(description="Take arguments to construct model")
parser.add_argument("unsupervised_file", type=str,
                    help="Relative path to file containing data for unsupervised training")
parser.add_argument("supervised_input_file", type=str,
                    help="Relative path to file containing the input data for supervised training")
parser.add_argument("output_file", type=str,
                    help="Relative path to file containing the output data for supervised training")

args = parser.parse_args()

gene_max = 10.0
gene_min = -10.0

unsupervised_load = np.loadtxt(open(args.unsupervised_file, "rb"), dtype=float, delimiter=",")
supervised_input_load = np.loadtxt(open(args.supervised_input_file, "rb"), dtype=float, delimiter=",")
supervised_output_load = np.loadtxt(open(args.output_file, "rb"), dtype=float, delimiter=",")

unsupervised_matrix = np.maximum(np.minimum(unsupervised_load - gene_min, gene_max - gene_min), 0) \
                      / (gene_max - gene_min)
supervised_input_matrix = np.maximum(np.minimum(supervised_input_load - gene_min, gene_max - gene_min), 0) \
                          / (gene_max - gene_min)
supervised_output_matrix = np.asarray(supervised_output_load)

input_size = len(list(supervised_input_matrix[0]))
output_size = 1


class MyDataset(Dataset):
    def __init__(self, inputs, outputs):
        super(MyDataset, self).__init__()

        self.inputs = [torch.from_numpy(np.atleast_1d(vec)).float() for vec in inputs]
        self.outputs = [torch.from_numpy(np.atleast_1d(vec)).float() for vec in outputs]

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

unsupervised_dataset = MyDataset(inputs=unsupervised_matrix, outputs=unsupervised_matrix)
supervised_dataset_biomass = MyDataset(inputs=supervised_input_matrix, outputs=[vec[0] for vec in supervised_output_matrix])
supervised_dataset_succinate = MyDataset(inputs=supervised_input_matrix, outputs=[vec[1] for vec in supervised_output_matrix])
supervised_dataset_ethanol = MyDataset(inputs=supervised_input_matrix, outputs=[vec[2] for vec in supervised_output_matrix])


def repeat(supervised_dataset, model_state_path):
    model = SimpleNetwork()
    model.cuda()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.save(model.state_dict(), model_state_path)

    def train(dataloader, epoch):
        model.train()
        train_loss = 0
        total_items = 0

        for batch_idx, (inputs, outputs) in enumerate(dataloader):
            inputs = inputs.cuda()
            outputs = outputs.cuda()

            optimizer.zero_grad()

            predictions = model(inputs)

            loss = criterion(predictions, outputs)

            train_loss += loss.item()
            total_items += len(predictions)

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print("Simple Network Train Epoch {} Loss {}".format(epoch, (train_loss / total_items) * 1e6))

    def test(dataloader):
        model.eval()

        with torch.no_grad():
            within_error = 0
            test_loss = 0
            total_items = 0

            for batch_idx, (inputs, outputs) in enumerate(dataloader):
                inputs = inputs.cuda()
                outputs = outputs.cuda()

                predictions = model(inputs)

                loss = criterion(predictions, outputs)

                for i in range(len(predictions)):
                    real = outputs[i].item()
                    pred = predictions[i].item()

                    if 1.1*real > pred > 0.9*real:
                        within_error += 1

                test_loss += loss.item()
                total_items += len(predictions)

            print("### Simple Network TEST Loss {}".format((test_loss / total_items) * 1e6))
            print("Within error: {} Total: {}".format(within_error, total_items))

            return test_loss/total_items, within_error/total_items

    indices = list(range(dataset.__len__()))
    shuffle(indices)

    train_indexes = np.array_split(indices, 10)

    test_result_loss = [0] * 10
    test_result_within_error = [0] * 10

    i = 0
    for test_idx in train_indexes:
        train_idx = list(set(indices) - set(test_idx))

        model.load_state_dict(torch.load(model_state_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset=supervised_dataset, batch_size=1000, sampler=train_sampler)
        test_loader = DataLoader(dataset=supervised_dataset, batch_size=100, sampler=test_sampler)

        for epoch in range(100):
            train(train_loader, epoch)

        test_result_loss[i], test_result_within_error[i] = test(test_loader)

        i += 1

    torch.save(model.state_dict(), model_state_path)

    return test_result_loss, test_result_within_error

if __name__ == "__main__":
    directory = "./outputs/simple_network"
    model_directory = "./models/simple_network"

    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    else:
        shutil.rmtree(directory, ignore_errors=True)

    if not os.path.exists("models"):
        os.mkdir("models")
    else:
        shutil.rmtree(model_directory, ignore_errors=True)

    os.mkdir(directory)
    os.mkdir(model_directory)

    sys.stdout = open("{}/Main.txt".format(directory), "w")

    phenotypes = ["biomass", "succinate", "ethanol"]
    datasets = [supervised_dataset_biomass, supervised_dataset_succinate, supervised_dataset_ethanol]

    for i in range(3):
        phenotype = phenotypes[i]
        dataset = datasets[i]

        loss_file = open("{}/{}_loss.csv".format(directory, phenotype), "w")
        error_file = open("{}/{}_within_error.csv".format(directory, phenotype), "w")

        loss_writer = csv.writer(loss_file)
        error_writer = csv.writer(error_file)

        for j in range(5):
            test_loss, test_within_error = repeat(dataset, "{}/{}_{}.pt".format(model_directory, phenotype, j))

            loss_writer.writerow(test_loss)
            error_writer.writerow(test_within_error)

        loss_file.close()
        error_file.close()
