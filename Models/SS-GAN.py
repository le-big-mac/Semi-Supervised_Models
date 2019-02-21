import torch
import os
import csv
from torch import nn
from torch.utils.data import DataLoader
from Models.utils import Arguments, KFoldSplits, Datasets, LoadData


class Classifier(nn.Module):
    def __init__(self, data_dim, hidden_dimensions, num_classes, activation):
        super(Classifier, self).__init__()

        dims = [data_dim] + hidden_dimensions

        layers = [
            nn.Sequential(
                nn.Linear(dims[i-1], dims[i]),
                activation,
            )
            for i in range(1, len(dims))
        ]

        self.fc_layers = nn.ModuleList(layers)
        self.classification_layer = nn.Linear(hidden_dimensions[-1], num_classes)
        self.discriminator_layer = nn.Sequential(
            nn.Linear(hidden_dimensions[-1], 1),
            nn.Sigmoid(),
        )

    # TODO: return features somehow
    # maybe don't have the separation of validity? K+1 classes
    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)

        return self.discriminator_layer(x), self.classification_layer(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dimensions, data_dim, activation):
        super(Generator, self).__init__()

        dims = [latent_dim] + hidden_dimensions

        layers = [
            nn.Sequential(
                nn.Linear(dims[i-1], dims[i]),
                activation,
            )
            for i in range(1, len(dims))
        ]

        self.fc_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dimensions[-1], data_dim)

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)

        return self.output_layer(x)


class SS_GAN:
    def __init__(self, gen_hidden_dimensions, dis_hidden_dimensions, data_size, latent_size, num_classes, activation,
                 device):
        self.G = Generator(latent_size, gen_hidden_dimensions, data_size, activation).to(device)
        self.D = Classifier(data_size, dis_hidden_dimensions, num_classes, activation).to(device)
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-3)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=1e-3)
        self.num_classes = num_classes
        self.device = device
        self.latent_dim = latent_size
        self.gen_state_path = 'state/ss-gen.pt'
        self.dis_state_path = 'state/ss-dis.pt'
        torch.save(self.G.state_dict(), self.gen_state_path)
        torch.save(self.D.state_dict(), self.dis_state_path)

    def classifier_loss(self, classification_logits, real_labels):
        # loss for (xi, yi) is -log(q(xi)(j)) where yi(j) == 1 (only one element in label vector)
        # safely ignore the fake label for real supervised loss
        return nn.CrossEntropyLoss(reduction='sum')(classification_logits, real_labels)

    def discriminator_real_loss(self, prob_fake):
        # labels are 0 because we want no probability for K+1 here
        return nn.BCELoss(reduction='sum')(prob_fake, torch.zeros(prob_fake.size()))

    def discriminator_fake_loss(self, prob_fake):
        # labels are 1 because discriminator should put all these in K+1)

        # Loss function if switch back to K+1 classes
        # probs = nn.Softmax(logits)
        # return nn.BCELoss(probs[:, self.num_classes], torch.ones(probs[:, 0].size()))

        return nn.BCELoss(reduction='sum')(prob_fake, torch.ones(prob_fake.size()))

    def simple_generator_loss(self, prob_fake):
        # only calculated for fake data batches
        # labels are 0 even though data is fake as loss for generator is higher when discriminator classifies correctly
        return nn.BCELoss(reduction='sum')(prob_fake, torch.zeros(prob_fake.size()))

    def feature_matching_loss(self, real_features, fake_features):
        # not sure why they do mean then do distance
        # L = ||E_real[f(x)] - E_fake[f(x)]||**2
        real_expectation = torch.mean(real_features, dim=0)
        fake_expectation = torch.mean(fake_features, dim=0)
        distance = torch.sum((real_expectation - fake_expectation) ** 2)

        return distance

    def train_one_epoch(self, epoch, supervised_dataloader, unsupervised_dataloader):
        total_supervised_loss = 0
        total_unsupervised_loss = 0
        total_gen_loss = 0

        supervised_samples = 0
        unsupervised_samples = 0
        gen_samples = 0

        self.D.train()
        self.G.train()

        # any additional batches in the supervised or unsupervised dataloader will be ignored
        # num_batches = min(supervised_batches, unsupervised_batches)
        for i, (labeled_data, unlabeled_data) in enumerate(zip(supervised_dataloader, unsupervised_dataloader)):
            # ----------------------
            # Discriminator training
            # ----------------------
            labeled_inputs, labeled_outputs = labeled_data

            labeled_inputs.to(self.device)
            labeled_outputs.to(self.device)
            unlabeled_data.to(self.device)

            self.D_optimizer.zero_grad()

            supervised_samples += len(labeled_inputs)

            _, labeled_pred = self.D(labeled_inputs)

            supervised_loss = self.classifier_loss(labeled_pred, labeled_outputs)
            supervised_loss.backward()

            total_supervised_loss += supervised_loss.item()

            unsupervised_samples += len(unlabeled_data)

            unsupervised_valid, _ = self.D(unlabeled_data.float())

            discriminator_real_loss = self.discriminator_real_loss(unsupervised_valid)
            discriminator_real_loss.backward()

            # Half of the data is fake (same amount as supervised+unsupervised)
            gen_input = torch.randn(len(unlabeled_data) + len(labeled_inputs), self.latent_dim, device=self.device)

            gen_samples += len(gen_input)

            fake = self.G(gen_input)

            fake_valid, _ = self.D(fake.detach())

            discriminator_fake_loss = self.discriminator_fake_loss(fake_valid)
            discriminator_fake_loss.backward()

            unsupervised_loss = discriminator_real_loss + discriminator_fake_loss

            total_unsupervised_loss += unsupervised_loss.item()

            self.D_optimizer.step()

            # ------------------
            # Generator training
            # ------------------

            self.G_optimizer.zero_grad()

            validity, _ = self.D(fake)

            generator_loss = self.simple_generator_loss(validity)
            generator_loss.backward()

            total_gen_loss += generator_loss.item()

            self.G_optimizer.step()

        print('Epoch: {} Supervised Loss: {} Unsupervised Loss: {} Gen Loss: {}'.
              format(epoch, total_supervised_loss/supervised_samples, total_unsupervised_loss/unsupervised_samples,
                     total_gen_loss / gen_samples))

        return total_supervised_loss/supervised_samples, total_unsupervised_loss/unsupervised_samples, \
               total_gen_loss/gen_samples

    def validation(self, supervised_dataloader):
        model = self.D

        model.eval()
        validation_loss = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(supervised_dataloader):
                data.to(self.device)
                labels.to(self.device)

                _, predictions = model(data)

                loss = self.classifier_loss(predictions, labels)

                validation_loss += loss.item()

        return validation_loss/len(supervised_dataloader.dataset)

    def test(self, dataloader):
        model = self.D

        model.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                _, outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        return correct/len(dataloader.dataset)

    def reset_model(self):
        self.G.load_state_dict(torch.load(self.gen_state_path))
        self.D.load_state_dict(torch.load(self.dis_state_path))

        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-3)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=1e-3)

    def full_train(self, unsupervised_dataset, train_dataset, validation_dataset=None):
        self.reset_model()

        # TODO: don't use arbitrary values for batch size
        unsupervised_dataloader = DataLoader(dataset=unsupervised_dataset, batch_size=180, shuffle=True)
        supervised_dataloader = DataLoader(dataset=train_dataset, batch_size=20, shuffle=True)

        if validation_dataset:
            validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_dataset.__len__())

        # simple early stopping employed (can change later)

        validation_result = float("inf")
        for epoch in range(50):

            self.train_one_epoch(epoch, supervised_dataloader, unsupervised_dataloader)

            if validation_dataset:
                val = self.validation(validation_dataloader)

                if val > validation_result:
                    break

                validation_result = val

    def full_test(self, test_dataset):
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())

        return self.test(test_dataloader)


if __name__ == '__main__':

    args = Arguments.parse_args()

    unsupervised_data, supervised_data, supervised_labels = LoadData.load_data_from_file(
        args.unsupervised_file, args.supervised_data_file, args.supervised_labels_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ss_gan = SS_GAN([200], [200], 500, 25, 10, nn.ReLU(), device)

    unsupervised_dataset = Datasets.UnsupervisedDataset(unsupervised_data)

    test_results = []

    for test_idx, train_idx in KFoldSplits.k_fold_splits(len(supervised_data), 10):

        train_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in train_idx],
                                                                 [supervised_labels[i] for i in train_idx])
        test_dataset = Datasets.SupervisedClassificationDataset([supervised_data[i] for i in test_idx],
                                                                [supervised_labels[i] for i in test_idx])

        ss_gan.full_train(unsupervised_data, train_dataset)

        correct_percentage = ss_gan.full_test(test_dataset)

        test_results.append(correct_percentage)

    if not os.path.exists('../results'):
        os.mkdir('../results')
        os.mkdir('../results/ss-gan')
    elif not os.path.exists('../results/ss-gan'):
        os.mkdir('../results/ss-gan')

    accuracy_file = open('../results/ss-gan/accuracy.csv', 'w')
    accuracy_writer = csv.writer(accuracy_file)

    accuracy_writer.writerow(test_results)
