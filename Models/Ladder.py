import math
import torch
from torch import nn
import torch.nn.functional as F
from itertools import cycle
from Models.Model import Model
from utils.trainingutils import EarlyStopping
import pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from statistics import mean


def bi(inits, size):
    return nn.Parameter(inits * torch.ones(size))


def wi(shape):
    return nn.Parameter(torch.randn(shape) / math.sqrt(shape[0]))


join = lambda l, u: torch.cat((l, u), 0)
labeled = lambda x, batch_size: x[:batch_size] if x is not None else x
unlabeled = lambda x, batch_size: x[batch_size:] if x is not None else x
split_lu = lambda x, batch_size: (labeled(x, batch_size), unlabeled(x, batch_size))


class encoders(nn.Module):
    def __init__(self, shapes, layer_sizes, L, device):
        super(encoders, self).__init__()
        self.W = nn.ParameterList([wi(s) for s in shapes])
        self.beta = nn.ParameterList([bi(0.0, s[1]) for s in shapes])
        self.gamma = nn.Parameter(bi(1.0, layer_sizes[-1]))
        self.batch_norm_clean_labelled = nn.ModuleList([nn.BatchNorm1d(s[1], affine=False) for s in shapes])
        self.batch_norm_clean_unlabelled = nn.ModuleList([nn.BatchNorm1d(s[1], affine=False) for s in shapes])
        self.batch_norm_noisy = nn.ModuleList([nn.BatchNorm1d(s[1], affine=False) for s in shapes])

        self.L = L
        self.device = device

    def forward(self, inputs, noise_std, training, batch_size):
        h = inputs + noise_std * torch.randn_like(inputs).to(self.device)  # add noise to input
        d = {}  # to store the pre-activation, activation, mean and variance for each layer
        # The data for labeled and unlabeled examples are stored separately
        d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h, batch_size)
        for l in range(1, self.L+1):
            # print("Layer ", l, ": ", layer_sizes[l-1], " -> ", layer_sizes[l])

            d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h, batch_size)
            z_pre = torch.mm(h, self.W[l-1])  # pre-activation

            if training:
                z_pre_l, z_pre_u = split_lu(z_pre, batch_size)  # split labeled and unlabeled examples
                m = z_pre_u.mean(dim=0)
                v = z_pre_u.var(dim=0)
                d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
                # Training batch normalization
                # batch normalization for labeled and unlabeled examples is performed separately
                if noise_std > 0:
                    # Corrupted encoder
                    # batch normalization + noise
                    z = join(self.batch_norm_noisy[l-1](z_pre_l), self.batch_norm_noisy[l-1](z_pre_u))
                    z += noise_std * torch.randn_like(z).to(self.device)
                else:
                    # Clean encoder
                    # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                    z = join(self.batch_norm_clean_labelled[l-1](z_pre_l),
                             self.batch_norm_clean_unlabelled[l-1](z_pre_u))

            else:
                # Evaluation batch normalization
                z = self.batch_norm_clean_labelled[l-1](z_pre)

            if l == self.L:
                # softmax done in nn.CrossEntropyLoss so output from model is linear
                h = self.gamma * (z + self.beta[l-1])
            else:
                # use ReLU activation in hidden layers
                h = F.relu(z + self.beta[l-1])

            d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z, batch_size)
        d['labeled']['h'][self.L], d['unlabeled']['h'][self.L] = split_lu(h, batch_size)
        return h, d


class decoders(nn.Module):
    def __init__(self, shapes, layer_sizes, L):
        super(decoders, self).__init__()

        self.V = nn.ParameterList([wi(s[::-1]) for s in shapes])

        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(size, affine=False) for size in layer_sizes])

        self.a1 = nn.ParameterList([bi(0., size) for size in layer_sizes])
        self.a2 = nn.ParameterList([bi(1., size) for size in layer_sizes])
        self.a3 = nn.ParameterList([bi(0., size) for size in layer_sizes])
        self.a4 = nn.ParameterList([bi(0., size) for size in layer_sizes])
        self.a5 = nn.ParameterList([bi(0., size) for size in layer_sizes])

        self.a6 = nn.ParameterList([bi(0., size) for size in layer_sizes])
        self.a7 = nn.ParameterList([bi(1., size) for size in layer_sizes])
        self.a8 = nn.ParameterList([bi(0., size) for size in layer_sizes])
        self.a9 = nn.ParameterList([bi(0., size) for size in layer_sizes])
        self.a10 = nn.ParameterList([bi(0., size) for size in layer_sizes])

        self.L = L

    def g_gauss(self, z_c, u, l):
        "gaussian denoising function proposed in the original paper"
        mu = self.a1[l] * torch.sigmoid(self.a2[l] * u + self.a3[l]) + self.a4[l] * u + self.a5[l]
        v = self.a6[l] * torch.sigmoid(self.a7[l] * u + self.a8[l]) + self.a9[l] * u + self.a10[l]

        z_est = (z_c - mu) * v + mu
        return z_est

    # Decoder
    def forward(self, y_c, corr, clean, batch_size):
        z_est = {}
        z_est_bn = {}
        for l in range(self.L, -1, -1):
            z_c = corr['unlabeled']['z'][l]
            if l == self.L:
                u = unlabeled(y_c, batch_size)
            else:
                u = torch.mm(z_est[l+1], self.V[l])
            u = self.batch_norm[l](u)
            z_est[l] = self.g_gauss(z_c, u, l)

            if l > 0:
                m = clean['unlabeled']['m'][l]
                v = clean['unlabeled']['v'][l]
                z_est_bn[l] = (z_est[l] - m) / torch.sqrt(v + 1e-10)
            else:
                z_est_bn[l] = z_est[l]

        return z_est_bn


class Ladder(nn.Module):
    def __init__(self, shapes, layer_sizes, L, device):
        super(Ladder, self).__init__()

        self.encoders = encoders(shapes, layer_sizes, L, device)
        self.decoders = decoders(shapes, layer_sizes, L)

    def forward_encoders(self, inputs, noise_std, train, batch_size):
        return self.encoders.forward(inputs, noise_std, train, batch_size)

    def forward_decoders(self, y_c, corr, clean, batch_size):
        return self.decoders.forward(y_c, corr, clean, batch_size)


class LadderNetwork(Model):
    def __init__(self, input_size, hidden_dimensions, num_classes, denoising_cost, lr, device, model_name, state_path):
        super(LadderNetwork, self).__init__(device, state_path, model_name)

        layer_sizes = [input_size] + hidden_dimensions + [num_classes]
        shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.L = len(layer_sizes) - 1
        self.ladder = Ladder(shapes, layer_sizes, self.L, device).to(device)
        self.optimizer = torch.optim.Adam(self.ladder.parameters(), lr=lr)
        self.supervised_cost_function = nn.CrossEntropyLoss()
        self.unsupervised_cost_function = nn.MSELoss(reduction='mean')

        self.denoising_cost = denoising_cost
        self.noise_std = 0.3

    def accuracy(self, dataloader, batch_size):
        self.ladder.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs, _ = self.ladder.forward_encoders(data, 0.0, False, batch_size)

                _, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
                correct += (predicted == labels).sum().item()

        return correct / len(dataloader.dataset)

    def train_ladder(self, max_epochs, supervised_dataloader, unsupervised_dataloader, validation_dataloader):
        epochs = []
        train_losses = []
        validation_accs = []

        early_stopping = EarlyStopping('{}/{}_inner.pt'.format(self.state_path, self.model_name))

        for epoch in range(max_epochs):
            if early_stopping.early_stop:
                break

            train_loss = 0
            for batch_idx, (labelled_data, unlabelled_data) in enumerate(
                    zip(cycle(supervised_dataloader), unsupervised_dataloader)):
                self.ladder.train()

                self.optimizer.zero_grad()

                labelled_images, labels = labelled_data
                labelled_images = labelled_images.to(self.device)
                labels = labels.to(self.device)

                unlabelled_images, _ = unlabelled_data
                unlabelled_images = unlabelled_images.to(self.device)

                inputs = torch.cat((labelled_images, unlabelled_images), 0)

                batch_size = labelled_images.size(0)

                y_c, corr = self.ladder.forward_encoders(inputs, self.noise_std, True, batch_size)
                y, clean = self.ladder.forward_encoders(inputs, 0.0, True, batch_size)

                z_est_bn = self.ladder.forward_decoders(F.softmax(y_c, dim=1), corr, clean, batch_size)

                cost = self.supervised_cost_function.forward(labeled(y_c, batch_size), labels)

                zs = clean['unlabeled']['z']

                u_cost = 0
                for l in range(self.L, -1, -1):
                    u_cost += self.unsupervised_cost_function.forward(z_est_bn[l], zs[l]) * self.denoising_cost[l]

                loss = cost + u_cost

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            if validation_dataloader is not None:
                acc = self.accuracy(validation_dataloader, 0)
                validation_accs.append(acc)
                early_stopping(1 - acc, self.ladder)

            epochs.append(epoch)
            train_losses.append(train_loss/len(unsupervised_dataloader))

        if validation_dataloader is not None:
            early_stopping.load_checkpoint(self.ladder)

        return epochs, train_losses, validation_accs

    def train_model(self, max_epochs, dataloaders):
        unsupervised_dataloader, supervised_dataloader, validation_dataloader = dataloaders

        epochs, losses, validation_accs = self.train_ladder(max_epochs, supervised_dataloader, unsupervised_dataloader,
                                                            validation_dataloader)

        return epochs, losses, validation_accs

    def test_model(self, test_dataloader):
        return self.accuracy(test_dataloader, 0)

    def classify(self, data):
        self.ladder.eval()

        return self.forward(data)

    def forward(self, data):
        y, _ = self.ladder.forward_encoders(data.to(self.device), 0.0, False, 0)

        return y


def hyperparameter_loop(fold, validation_fold, state_path, results_path, dataloaders, input_size,
                        num_classes, max_epochs, device):
    hidden_layer_size = min(500, (input_size + num_classes) // 2)
    hidden_layers = range(1, 5)
    unsupervised, supervised, validation, test = dataloaders
    train_dataloaders = (unsupervised, supervised, validation)
    num_labelled = len(supervised.dataset)
    lr = 1e-3

    best_acc = 0
    best_params = None

    logging_list = []
    hyperparameter_file = '{}/{}_{}_{}_hyperparameters.p'.format(results_path, fold, validation_fold, num_labelled)
    pickle.dump(logging_list, open(hyperparameter_file, 'wb'))

    for h in hidden_layers:
        print('Ladder hidden layers {}'.format(h))
        logging_list = pickle.load(open(hyperparameter_file, 'rb'))

        denoising_cost = [1000.0, 10.0] + ([0.1] * h)

        model_name = '{}_{}_{}_{}'.format(fold, validation_fold, num_labelled, h)
        model = LadderNetwork(input_size, [hidden_layer_size] * h, num_classes, denoising_cost, lr, device, model_name,
                              state_path)

        epochs, losses, val_accs = model.train_model(max_epochs, train_dataloaders)
        validation_result = model.test_model(validation)

        model_path = '{}/{}.pt'.format(state_path, model_name)
        torch.save(model.state_dict(), model_path)

        params = {'model name': model_name, 'input size': input_size, 'hidden layers': h * [hidden_layer_size],
                  'num classes': num_classes}
        logging = {'params': params, 'model name': model_name, 'accuracy': validation_result, 'epochs': epochs,
                   'losses': losses, 'accuracies': val_accs}

        logging_list.append(logging)
        pickle.dump(logging_list, open(hyperparameter_file, 'wb'))

        if validation_result > best_acc:
            best_acc = validation_result
            best_params = params

        if device == 'cuda':
            torch.cuda.empty_cache()

    model_name = best_params['model name']
    hidden_layers = best_params['hidden layers']
    denoising_cost = [1000.0, 10.0] + ([0.1] * len(hidden_layers))
    model = LadderNetwork(input_size, hidden_layers, num_classes, denoising_cost, lr, device, model_name, state_path)
    model.load_state_dict(torch.load('{}/{}.pt'.format(state_path, model_name)))
    test_acc = model.test_model(test)
    classify = model.classify(test.dataset.tensors[0])

    return model_name, test_acc, classify


def tool_hyperparams(train_val_folds, labelled_data, labels, unlabelled_data, output_folder, device):
    input_size = labelled_data.size(1)
    num_classes = labels.unique().size(0)
    state_path = '{}/state'.format(output_folder)

    hidden_layer_size = min(500, (input_size + num_classes) // 2)
    hidden_layers = range(1, 5)
    lr = 1e-3

    best_accuracies = [0, 0]
    best_params = None

    normalizer = StandardScaler()
    all_data = torch.tensor(normalizer.fit_transform(torch.cat((labelled_data, unlabelled_data)).numpy())).float()
    labelled_data = all_data[:len(labels)]

    for h in hidden_layers:
        print('Ladder params {}'.format(h))

        model_name = '{}'.format(h)
        denoising_cost = [1000.0, 10.0] + ([0.1] * h)
        params = {'model name': model_name, 'input size': input_size, 'hidden layers': h * [hidden_layer_size],
                  'denoising cost': denoising_cost, 'num classes': num_classes}

        accuracies = []

        for train_ind, val_ind in train_val_folds:
            s_d = TensorDataset(labelled_data[train_ind], labels[train_ind])

            unlabelled_data = torch.cat((all_data[len(labels):], labelled_data[train_ind]))
            u_d = TensorDataset(unlabelled_data, -1 * torch.ones(unlabelled_data.size(0)))
            v_d = TensorDataset(labelled_data[val_ind], labels[val_ind])

            s_dl = DataLoader(s_d, batch_size=100, shuffle=True)
            u_dl = DataLoader(u_d, batch_size=100, shuffle=True)
            v_dl = DataLoader(v_d, batch_size=v_d.__len__())

            model = LadderNetwork(input_size, [hidden_layer_size] * h, num_classes, denoising_cost, lr,
                                  device, model_name, state_path)
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
    unlabelled_data = all_data
    u_d = TensorDataset(unlabelled_data, -1 * torch.ones(unlabelled_data.size(0)))

    s_dl = DataLoader(s_d, batch_size=100, shuffle=True)
    u_dl = DataLoader(u_d, batch_size=100, shuffle=True)

    final_model = LadderNetwork(best_params['input size'], best_params['hidden layers'], best_params['num classes'],
                                best_params['denoising cost'], lr, device, 'ladder', state_path)
    final_model.train_model(100, (u_dl, s_dl, None))

    return final_model, normalizer, best_accuracies
