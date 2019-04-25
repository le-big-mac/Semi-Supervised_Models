import torch
from torch import nn
from torch.nn import functional as F
from Models.BuildingBlocks import VAE, Classifier
from Models.Model import Model
from utils.trainingutils import EarlyStopping, unsupervised_validation_loss
import pickle


class M1(Model):
    def __init__(self, input_size, hidden_dimensions_encoder, latent_size, hidden_dimensions_classifier,
                 num_classes, output_activation, lr, dataset_name, device, model_name, state_path):
        super(M1, self).__init__(dataset_name, device)

        self.VAE = VAE(input_size, hidden_dimensions_encoder, latent_size, output_activation).to(device)
        self.VAE_optim = torch.optim.Adam(self.VAE.parameters(), lr=lr)
        self.Encoder = self.VAE.encoder

        self.Classifier = Classifier(latent_size, hidden_dimensions_classifier, num_classes).to(device)
        self.Classifier_criterion = nn.CrossEntropyLoss()
        self.Classifier_optim = torch.optim.Adam(self.Classifier.parameters(), lr=lr)

        self.state_path = state_path
        self.model_name = model_name

    def VAE_criterion(self, batch_params, x):
        # KL divergence between two normal distributions (N(0, 1) and parameterized)
        recons, mu, logvar = batch_params

        KLD = 0.5*torch.sum(logvar.exp() + mu.pow(2) - logvar - 1, dim=1)

        # reconstruction error (use BCE because we normalize input data to [0, 1] and sigmoid output)
        recons = F.mse_loss(recons, x, reduction='none').sum(dim=1)

        # BCE used for mnist in original, removed here to allow for non-constrained input
        # recons = F.binary_cross_entropy(recons, x, reduction='none').sum(dim=1)

        return (KLD + recons).mean()

    def train_VAE(self, max_epochs, train_dataloader, validation_dataloader):
        early_stopping = EarlyStopping('{}/{}_autoencoder.pt'.format(self.state_path, self.model_name), patience=10)

        for epoch in range(max_epochs):
            if early_stopping.early_stop:
                break

            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_dataloader):
                self.VAE.train()

                data = data.to(self.device)

                self.VAE_optim.zero_grad()

                batch_params = self.VAE(data)

                loss = self.VAE_criterion(batch_params, data)

                train_loss += loss.item()

                loss.backward()
                print(loss.item())
                self.VAE_optim.step()

            validation_loss = unsupervised_validation_loss(self.VAE, validation_dataloader, self.VAE_criterion,
                                                           self.device)

            early_stopping(validation_loss, self.VAE)

            # print('Unsupervised Epoch: {} Loss: {} Validation loss: {}'.format(epoch, train_loss, validation_loss))

        if early_stopping.early_stop:
            early_stopping.load_checkpoint(self.VAE)

    def train_classifier(self, max_epochs, train_dataloader, validation_dataloader, comparison):
        epochs = []
        train_losses = []
        validation_accs = []

        early_stopping = EarlyStopping('{}/{}_classifier.pt'.format(self.state_path, self.model_name))

        for epoch in range(max_epochs):
            if early_stopping.early_stop:
                break

            train_loss = 0
            for batch_idx, (data, labels) in enumerate(train_dataloader):
                self.Classifier.train()

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.Classifier_optim.zero_grad()

                with torch.no_grad():
                    z, _, _ = self.Encoder(data)

                pred = self.Classifier(z)

                loss = self.Classifier_criterion(pred, labels)

                loss.backward()
                self.Classifier_optim.step()

                if comparison:
                    acc = self.accuracy(validation_dataloader)

                    epochs.append(epoch)
                    train_losses.append(loss.item())
                    validation_accs.append(acc)

                    early_stopping(1 - acc, self.Classifier)
                else:
                    train_loss += loss.item()

            if not comparison:
                acc = self.accuracy(validation_dataloader)

                epochs.append(epoch)
                train_losses.append(train_loss/len(train_dataloader))
                validation_accs.append(acc)

                early_stopping(1 - acc, self.Classifier)
                # print('Supervised Epoch: {} Validation acc: {}'.format(epoch, val))

        early_stopping.load_checkpoint(self.Classifier)

        return epochs, train_losses, validation_accs

    def accuracy(self, dataloader):
        self.Encoder.eval()
        self.Classifier.eval()

        correct = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                z, _, _ = self.Encoder(data)
                outputs = self.Classifier(z)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        return correct / len(dataloader.dataset)

    def train_model(self, max_epochs, dataloaders, comparison):
        unsupervised_dataloader, supervised_dataloader, validation_dataloader = dataloaders

        self.train_VAE(max_epochs, unsupervised_dataloader, validation_dataloader)

        classifier_epochs, classifier_losses, classifier_accs = \
            self.train_classifier(max_epochs, supervised_dataloader, validation_dataloader, comparison)

        return classifier_epochs, classifier_losses, classifier_accs

    def test_model(self, test_dataloader):
        return self.accuracy(test_dataloader)

    def classify(self, data):
        self.Encoder.eval()
        self.Classifier.eval()

        return self.forward(data)

    def forward(self, data):
        z, _, _ = self.Encoder(data)

        return self.Classifier(z)


def hyperparameter_loop(fold, state_path, results_path, dataset_name, dataloaders, input_size, num_classes, max_epochs,
                        device):
    hidden_layer_vae_size = min(500, (input_size + num_classes) // 2)
    hidden_layer_classifier_size = 50
    hidden_layers_vae = range(1, 3)
    hidden_layers_classifier = range(0, 2)
    z_size = [200, 100, 50]
    param_combinations = [(i, j, k) for i in hidden_layers_vae for j in hidden_layers_classifier for k in z_size]
    lr = 1e-3

    unsupervised, supervised, validation, test = dataloaders
    train_dataloaders = (unsupervised, supervised, validation)
    num_labelled = len(supervised.dataset)

    best_acc = 0
    best_params = None

    logging_list = []
    hyperparameter_file = '{}/{}_{}_hyperparameters.p'.format(results_path, fold, num_labelled)
    pickle.dump(logging_list, open(hyperparameter_file, 'wb'))

    for p in param_combinations:
        print('M1 params {}'.format(p))
        logging_list = pickle.load(open(hyperparameter_file, 'rb'))

        h_v, h_c, z = p

        model_name = '{}_{}_{}_{}_{}'.format(fold, num_labelled, h_v, h_c, z)
        model = M1(input_size, h_v * [hidden_layer_vae_size], z, h_c * [hidden_layer_classifier_size], num_classes,
                   lambda x: x, lr, dataset_name, device, model_name, state_path)
        epochs, losses, val_accs = model.train_model(max_epochs, train_dataloaders, False)
        validation_result = model.test_model(validation)

        model_path = '{}/{}.pt'.format(state_path, model_name)
        torch.save(model.state_dict(), model_path)

        params = {'model name': model_name, 'input size': input_size, 'hidden layers vae': h_v * [hidden_layer_vae_size],
                  'hidden layers classifier': h_c * [hidden_layer_classifier_size], 'latent dim': z,
                  'num classes': num_classes}
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
    model = M1(input_size, hidden_v, latent, hidden_c, num_classes, lambda x: x, lr, dataset_name, device, model_name,
               state_path)
    model.load_state_dict(torch.load('{}/{}.pt'.format(state_path, model_name)))
    test_acc = model.test_model(test)

    return model_name, test_acc
