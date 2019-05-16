import torch
import pickle
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle
from Models.BuildingBlocks import VariationalEncoder, Decoder, Classifier
from Models.Model import Model
from utils.trainingutils import EarlyStopping
from statistics import mean
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------
# Implementation of Kingma M2 semi-supervised variational autoencoder
# -----------------------------------------------------------------------


class VAE_M2(nn.Module):
    def __init__(self, input_size, hidden_dimensions_encoder, hidden_dimensions_decoder, latent_dim, num_classes,
                 output_activation):
        super(VAE_M2, self).__init__()

        self.encoder = VariationalEncoder(input_size + num_classes, hidden_dimensions_encoder, latent_dim)
        self.decoder = Decoder(input_size, hidden_dimensions_decoder, latent_dim + num_classes, output_activation)

    def forward(self, x, y):
        z, mu, logvar = self.encoder(torch.cat((x, y), dim=1))

        out = self.decoder(torch.cat((z, y), dim=1))

        return out, mu, logvar


class M2(nn.Module):
    def __init__(self, input_size, hidden_dimensions_VAE, hidden_dimensions_clas, latent_dim, num_classes,
                 output_activation):
        super(M2, self).__init__()

        self.VAE = VAE_M2(input_size, hidden_dimensions_VAE, hidden_dimensions_VAE, latent_dim, num_classes,
                          output_activation)
        self.Classifier = Classifier(input_size, hidden_dimensions_clas, num_classes)

    def classify(self, x):
        return self.Classifier(x)

    def forward(self, x, y):
        return self.VAE(x, y)


class M2Runner(Model):
    def __init__(self, input_size, hidden_dimensions_VAE, hidden_dimensions_clas, latent_dim, num_classes, activation,
                 lr, device, model_name, state_path):
        super(M2Runner, self).__init__(device, state_path, model_name)

        self.M2 = M2(input_size, hidden_dimensions_VAE, hidden_dimensions_clas, latent_dim,
                     num_classes, activation).to(device)
        # change this to something more applicable with softmax
        self.optimizer = torch.optim.Adam(self.M2.parameters(), lr=lr)
        self.num_classes = num_classes

    def onehot(self, labels):
        labels = labels.unsqueeze(1)

        y = torch.zeros(labels.size(0), self.num_classes).to(self.device)
        y = y.scatter(1, labels, 1)

        return y

    def minus_L(self, x, recons, mu, logvar, y):
        # KL divergence between two normal distributions (N(0, 1) and parameterized)
        KLD = 0.5*torch.sum(logvar.exp() + mu.pow(2) - logvar - 1, dim=1)

        # reconstruction error (use BCE because we normalize input data to [0, 1] and sigmoid output)
        accuracy = -F.binary_cross_entropy(recons, x, reduction='none').sum(dim=1)
        # accuracy = -F.mse_loss(recons, x, reduction='none').sum(dim=1)

        # prior over y (commented out because a uniform prior results in a constant for all labels)
        # prior_y = log_standard_categorical(y)

        return accuracy - KLD

    def log_standard_categorical(self, y):
        # this is useless when the data is uniformly distributed as it returns a constant
        prior = torch.ones((y.size(0), self.num_classes), requires_grad=False)

        return -F.cross_entropy(prior, y, reduction='none')

    def make_labels(self, batch_size):
        labels = []
        for i in range(self.num_classes):
            labels.append(i * torch.ones(batch_size).to(self.device).long())

        labels = torch.cat(labels)

        return labels

    def minus_U(self, x, pred_y):
        # gives probability for each label
        logits = F.softmax(pred_y, dim=1)

        y = self.make_labels(x.size(0))
        y_onehot = self.onehot(y)
        x = x.repeat(self.num_classes, 1)

        recons, mu, logvar = self.M2(x, y_onehot)

        minus_L = self.minus_L(x, recons, mu, logvar, y)
        minus_L = minus_L.view_as(logits.t()).t()

        minus_L = (logits * minus_L).sum(dim=1)

        H = self.H(logits)

        minus_U = H + minus_L

        return minus_U.mean()

    def H(self, logits):
        return -torch.sum(logits * torch.log(logits + 1e-8), dim=1)

    def elbo(self, x, y=None):
        if y is not None:
            recons, mu, logvar = self.M2(x, self.onehot(y))

            return -self.minus_L(x, recons, mu, logvar, y).mean()

        else:
            pred_y = self.M2.classify(x)

            return -self.minus_U(x, pred_y)

    def train_m2(self, max_epochs, labelled_loader, unlabelled_loader, validation_loader):

        if unlabelled_loader is None:
            alpha = 1
        else:
            alpha = 0.1 * len(unlabelled_loader.dataset)/len(labelled_loader.dataset)

        epochs = []
        train_losses = []
        validation_accs = []

        early_stopping = EarlyStopping('{}/{}_inner.pt'.format(self.state_path, self.model_name))

        for epoch in range(max_epochs):
            if early_stopping.early_stop:
                break

            if unlabelled_loader is not None:
                data_iterator = zip(cycle(labelled_loader), unlabelled_loader)
            else:
                data_iterator = zip(labelled_loader, cycle([None]))

            train_loss = 0
            for batch_idx, (labelled_data, unlabelled_data) in enumerate(data_iterator):
                self.M2.train()
                self.optimizer.zero_grad()

                labelled_images, labels = labelled_data
                labelled_images = labelled_images.float().to(self.device)
                labels = labels.to(self.device)

                labelled_predictions = self.M2.classify(labelled_images)
                labelled_loss = F.cross_entropy(labelled_predictions, labels)

                # labelled images ELBO
                L = self.elbo(labelled_images, y=labels)

                loss = L + alpha*labelled_loss

                if unlabelled_data is not None:
                    unlabelled_images, _ = unlabelled_data
                    unlabelled_images = unlabelled_images.float().to(self.device)

                    U = self.elbo(unlabelled_images)

                    loss += U

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            if validation_loader is not None:
                acc = self.accuracy(validation_loader)
                validation_accs.append(acc)
                early_stopping(1 - acc, self.M2)

            epochs.append(epoch)
            train_losses.append(train_loss)

        if validation_loader is not None:
            early_stopping.load_checkpoint(self.M2)

        return epochs, train_losses, validation_accs

    def accuracy(self, dataloader):
        self.M2.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.float().to(self.device)
                labels = labels.to(self.device)

                outputs = self.M2.classify(data)

                _, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
                correct += (predicted == labels).sum().item()

        return correct / len(dataloader.dataset)

    def train_model(self, max_epochs, dataloaders):
        unsupervised_dataloader, supervised_dataloader, validation_dataloader = dataloaders

        epochs, losses, validation_accs = self.train_m2(max_epochs, supervised_dataloader, unsupervised_dataloader,
                                                        validation_dataloader)

        return epochs, losses, validation_accs

    def test_model(self, test_dataloader):
        return self.accuracy(test_dataloader)

    def classify(self, data):
        self.M2.eval()

        return self.forward(data)

    def forward(self, data):
        return self.M2.classify(data.to(self.device))


def hyperparameter_loop(fold, validation_fold, state_path, results_path, dataloaders, input_size,
                        num_classes, max_epochs, device):
    hidden_layer_size = min(500, (input_size + num_classes) // 2)
    hidden_layers_vae = range(1, 3)
    hidden_layers_classifier = range(1, 3)
    z_size = [200, 100, 50]
    param_combinations = [(i, j, k) for i in hidden_layers_vae for j in hidden_layers_classifier for k in z_size]
    lr = 1e-3

    unsupervised, supervised, validation, test = dataloaders
    train_dataloaders = (unsupervised, supervised, validation)
    num_labelled = len(supervised.dataset)

    best_acc = 0
    best_params = None

    logging_list = []
    hyperparameter_file = '{}/{}_{}_{}_hyperparameters.p'.format(results_path, fold, validation_fold, num_labelled)
    pickle.dump(logging_list, open(hyperparameter_file, 'wb'))

    for p in param_combinations:
        print('M2 params {}'.format(p))
        logging_list = pickle.load(open(hyperparameter_file, 'rb'))

        h_v, h_c, z = p

        model_name = '{}_{}_{}_{}_{}_{}'.format(fold, validation_fold, num_labelled, h_v, h_c, z)
        model = M2Runner(input_size, [hidden_layer_size] * h_v, [hidden_layer_size] * h_c, z, num_classes,
                         nn.Sigmoid(), lr, device, model_name, state_path)
        epochs, losses, val_accs = model.train_model(max_epochs, train_dataloaders)
        validation_result = model.test_model(validation)

        model_path = '{}/{}.pt'.format(state_path, model_name)
        torch.save(model.state_dict(), model_path)

        params = {'model name': model_name, 'input size': input_size, 'hidden layers vae': h_v * [hidden_layer_size],
                  'hidden layers classifier': h_c * [hidden_layer_size], 'latent dim': z, 'num classes': num_classes}
        logging = {'params': params, 'filepath': model_path, 'accuracy': validation_result, 'epochs': epochs,
                   'losses': losses, 'accuracies': val_accs}

        logging_list.append(logging)
        pickle.dump(logging_list, open(hyperparameter_file, 'wb'))

        if validation_result > best_acc:
            best_acc = validation_result
            best_params = params

        if device == 'cuda':
            torch.cuda.empty_cache()

    model_name = best_params['model name']
    hidden_v = best_params['hidden layers vae']
    hidden_c = best_params['hidden layers classifier']
    latent = best_params['latent dim']
    model = M2Runner(input_size, hidden_v, hidden_c, latent, num_classes, nn.Sigmoid(), lr, device,
                     model_name, state_path)
    model.load_state_dict(torch.load('{}/{}.pt'.format(state_path, model_name)))
    test_acc = model.test_model(test)
    classify = model.classify(test.dataset.tensors[0])

    return model_name, test_acc, classify


def tool_hyperparams(train_val_folds, labelled_data, labels, unlabelled_data, output_folder, device):
    input_size = labelled_data.size(1)
    num_classes = labels.unique().size(0)
    state_path = '{}/state'.format(output_folder)

    hidden_layer_size = min(500, (input_size + num_classes) // 2)
    hidden_layers_vae = range(1, 3)
    hidden_layers_classifier = range(1, 3)
    z_size = [200, 100, 50]
    param_combinations = [(i, j, k) for i in hidden_layers_vae for j in hidden_layers_classifier for k in z_size]
    lr = 1e-3

    best_accuracies = [0, 0]
    best_params = None

    normalizer = MinMaxScaler()
    data = torch.tensor(normalizer.fit_transform(torch.cat((labelled_data, unlabelled_data)).numpy())).float()
    labelled_data = data[:len(labels)]
    unlabelled_data = data[len(labels):]

    for p in param_combinations:
        print('M2 params {}'.format(p))
        h_v, h_c, z = p
        model_name = '{}_{}_{}'.format(h_v, h_c, z)
        params = {'model name': model_name, 'input size': input_size, 'hidden layers vae': h_v * [hidden_layer_size],
                  'hidden layers classifier': h_c * [hidden_layer_size], 'latent dim': z, 'num classes': num_classes}

        accuracies = []

        for train_ind, val_ind in train_val_folds:
            s_d = TensorDataset(labelled_data[train_ind], labels[train_ind])
            v_d = TensorDataset(labelled_data[val_ind], labels[val_ind])

            s_dl = DataLoader(s_d, batch_size=100, shuffle=True)
            v_dl = DataLoader(v_d, batch_size=v_d.__len__())

            if len(unlabelled_data) == 0:
                u_dl = None
            else:
                u_d = TensorDataset(unlabelled_data, -1 * torch.ones(unlabelled_data.size(0)))
                u_dl = DataLoader(u_d, batch_size=100, shuffle=True)

            model = M2Runner(input_size, [hidden_layer_size] * h_v, [hidden_layer_size] * h_c, z, num_classes,
                             nn.Sigmoid(), lr, device, model_name, state_path)
            model.train_model(100, (u_dl, s_dl, v_dl))
            validation_result = model.test_model(v_dl)
            print('Validation accuracy: {}'.format(validation_result))

            accuracies.append(validation_result)

            if device == 'cuda':
                torch.cuda.empty_cache()

        if mean(accuracies) > mean(best_accuracies):
            best_accuracies = accuracies
            best_params = params

    s_d = TensorDataset(labelled_data, labels)
    s_dl = DataLoader(s_d, batch_size=100, shuffle=True)

    if len(unlabelled_data) == 0:
        u_dl = None
    else:
        u_d = TensorDataset(unlabelled_data, -1 * torch.ones(unlabelled_data.size(0)))
        u_dl = DataLoader(u_d, batch_size=100, shuffle=True)

    final_model = M2Runner(best_params['input size'], best_params['hidden layers vae'], best_params['hidden layers classifier'],
                           best_params['latent dim'], best_params['num classes'], nn.Sigmoid(), lr, device, 'm2', state_path)
    final_model.train_model(100, (u_dl, s_dl, None))

    return final_model, normalizer, best_accuracies
