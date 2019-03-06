import torch
from torch import nn
from itertools import cycle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Models.Ladder.Encoder import StackedEncoders
from Models.Ladder.Decoder import StackedDecoders


class Ladder(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes, activation_strings, noise_level, device):
        super(Ladder, self).__init__()

        self.Classifier = StackedEncoders(input_size, hidden_dimensions, num_classes, activation_strings, noise_level,
                                          device)

        self.StackedDecoders = StackedDecoders(num_classes, hidden_dimensions[::-1], input_size, device)

        self.device = device

    def forward_encoders_clean(self, x):
        return self.Classifier.forward_clean(x)

    def forward_encoders_noise(self, x):
        return self.Classifier.forward_noisy(x)

    def forward_decoders(self, h_L, z_tildes, z_pre_layers):
        return self.StackedDecoders.forward(h_L, z_tildes, z_pre_layers)


class LadderNetwork:
    def __init__(self, input_size, hidden_dimensions, num_classes, activation_strings, noise_level,
                 unsupervised_loss_multipliers, device):

        self.Ladder = Ladder(input_size, hidden_dimensions, num_classes, activation_strings, noise_level, device)
        self.optimizer = torch.optim.Adam(self.Ladder.parameters(), lr=0.02)

        self.loss_unsupervised = nn.MSELoss()
        self.loss_supervised = nn.CrossEntropyLoss()

        self.unsupervised_multipliers = unsupervised_loss_multipliers

        self.device = device

    def train_one_epoch(self, epoch, labelled_loader, unlabelled_loader, validation_loader):
        ladder = self.Ladder

        for batch_idx, (labelled, unlabelled_data) in enumerate(zip(cycle(labelled_loader), unlabelled_loader)):
            ladder.train()

            labelled_data, labelled_target = labelled

            unlabelled_data = unlabelled_data.float().to(self.device)
            labelled_data = labelled_data.float().to(self.device)
            labelled_target = labelled_target.to(self.device)

            self.optimizer.zero_grad()

            # do a noisy pass for labelled data
            output_noise_labelled, tilde_z_layers_labelled = ladder.forward_encoders_noise(labelled_data)

            # N.B. in the original paper it says to calculate the denoising cost for labelled examples
            # but the code that goes with the paper does not
            # Instead the code has labeled examples included in the unlabeled examples as well (avoids overtraining
            # on the labeled examples)

            # # do a clean pass for labelled data
            # output_clean_labelled, z_layers_labelled, z_pre_layers_labelled = \
            #     ladder.forward_encoders_clean(labelled_data)
            #
            # # pass through decoders
            # hat_z_layers_labelled, bn_hat_z_layers_labelled = \
            #     ladder.forward_decoders(F.softmax(output_noise_labelled), tilde_z_layers_labelled,
            #                             z_pre_layers_labelled)

            # do a noisy pass for unlabelled_data
            output_noise_unlabelled, tilde_z_layers_unlabelled  = ladder.forward_encoders_noise(unlabelled_data)

            # do a clean pass for unlabelled data
            output_clean_unlabelled, z_layers_unlabelled, z_pre_layers_unlabelled = \
                ladder.forward_encoders_clean(unlabelled_data)

            # pass through decoders
            hat_z_layers_unlabelled, bn_hat_z_layers_unlabelled = \
                ladder.forward_decoders(F.softmax(output_noise_unlabelled), tilde_z_layers_unlabelled,
                                        z_pre_layers_unlabelled)

            # calculate costs
            cost_supervised = self.loss_supervised.forward(output_noise_labelled, labelled_target)
            cost_unsupervised = 0.
            assert len(z_layers_unlabelled) == len(bn_hat_z_layers_unlabelled)
            for cost_lambda, z, bn_hat_z in zip(self.unsupervised_multipliers, z_layers_unlabelled,
                                                bn_hat_z_layers_unlabelled):
                c = cost_lambda * self.loss_unsupervised.forward(bn_hat_z, z)
                cost_unsupervised += c

            # original paper has this cost but code does not
            # for cost_lambda, z, bn_hat_z in zip(self.unsupervised_multipliers, z_layers_labelled,
            #                                     bn_hat_z_layers_labelled):
            #     c = cost_lambda * self.loss_unsupervised.forward(bn_hat_z, z)
            #     cost_unsupervised += c

            # backprop
            cost = cost_supervised + cost_unsupervised
            cost.backward()
            self.optimizer.step()

            print('Epoch: {} Supervised Cost: {:.4f} Unsupervised Cost: {:.4f} Validation Acc: {:.4f}'
                  .format(epoch, cost_supervised.item(), cost_unsupervised.item(), self.accuracy(validation_loader)))

    def accuracy(self, dataloader):
        self.Ladder.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.float().to(self.device)
                labels = labels.to(self.device)

                outputs, _, _ = self.Ladder.forward_encoders_clean(data)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        return correct / len(dataloader.dataset)

    def full_train(self, unsupervised_dataset, supervised_train_dataset, validation_dataset):

        labelled_loader = DataLoader(supervised_train_dataset, batch_size=100, shuffle=True)
        unlabelled_loader = DataLoader(unsupervised_dataset, batch_size=100, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=validation_dataset.__len__())

        for epoch in range(100):
            self.train_one_epoch(epoch, labelled_loader, unlabelled_loader, validation_loader)

    def test(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__())

        return self.accuracy(test_loader)
