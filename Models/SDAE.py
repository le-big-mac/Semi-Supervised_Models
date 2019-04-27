import torch
import csv
from torch import nn
from utils.trainingutils import accuracy
from Models.BuildingBlocks import Encoder, Decoder
from Models.Model import Model
from utils.trainingutils import EarlyStopping
import pickle


class AutoencoderSDAE(nn.Module):
    def __init__(self, encoder):
        super(AutoencoderSDAE, self).__init__()

        self.encoder = encoder
        self.decoder = Decoder(encoder.latent.in_features, [], encoder.latent.out_features, lambda x: x)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y


class SDAEClassifier(nn.Module):
    def __init__(self, input_size, hidden_dimensions, num_classes):
        super(SDAEClassifier, self).__init__()

        dims = [input_size] + hidden_dimensions

        layers = [Encoder(dims[i], [], dims[i+1], nn.ReLU())
                  for i in range(0, len(dims)-1)]

        self.hidden_layers = nn.ModuleList(layers)
        self.classification_layer = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        return self.classification_layer(x)


class SDAE(Model):
    def __init__(self, input_size, hidden_dimensions, num_classes, lr, dataset_name, device, model_name, state_path):
        super(SDAE, self).__init__(dataset_name, device)

        self.SDAEClassifier = SDAEClassifier(input_size, hidden_dimensions, num_classes).to(device)
        self.optimizer = torch.optim.Adam(self.SDAEClassifier.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.state_path = state_path
        self.model_name = model_name

    def pretrain_hidden_layers(self, max_epochs, pretraining_dataloader):
        for i in range(len(self.SDAEClassifier.hidden_layers)):
            dae = AutoencoderSDAE(self.SDAEClassifier.hidden_layers[i]).to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(dae.parameters(), lr=1e-3)

            previous_layers = self.SDAEClassifier.hidden_layers[0:i]

            # TODO: think about implementing early stopping
            for epoch in range(50):
                for batch_idx, (data, _) in enumerate(pretraining_dataloader):
                    dae.train()
                    data = data.to(self.device)

                    with torch.no_grad():
                        for layer in previous_layers:
                            data = layer(data)

                    noisy_data = data.add(0.3 * torch.randn_like(data).to(self.device))

                    optimizer.zero_grad()

                    predictions = dae(noisy_data)

                    loss = criterion(predictions, data)
                    print(loss.item())

                    loss.backward()
                    optimizer.step()

                    # print('Unsupervised Layer: {} Epoch: {} Loss: {}'.format(i, epoch, loss.item()))

    def train_classifier(self, max_epochs, train_dataloader, validation_dataloader, comparison):
        epochs = []
        train_losses = []
        validation_accs = []

        early_stopping = EarlyStopping('{}/{}_inner.pt'.format(self.state_path, self.model_name))

        for epoch in range(max_epochs):
            if early_stopping.early_stop:
                break

            train_loss = 0
            for batch_idx, (data, labels) in enumerate(train_dataloader):
                self.SDAEClassifier.train()

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                predictions = self.SDAEClassifier(data)

                loss = self.criterion(predictions, labels)

                loss.backward()
                self.optimizer.step()

                if comparison:
                    acc = accuracy(self.SDAEClassifier, validation_dataloader, self.device)
                    print(acc)

                    epochs.append(epoch)
                    train_losses.append(loss.item())
                    validation_accs.append(acc)

                    early_stopping(1 - acc, self.SDAEClassifier)
                else:
                    train_loss += loss.item()

            if not comparison:
                acc = accuracy(self.SDAEClassifier, validation_dataloader, self.device)

                epochs.append(epoch)
                train_losses.append(train_loss/len(train_dataloader))
                validation_accs.append(acc)

                early_stopping(1 - acc, self.SDAEClassifier)

        early_stopping.load_checkpoint(self.SDAEClassifier)

        return epochs, train_losses, validation_accs

    def train_model(self, max_epochs, dataloaders, comparison):
        unsupervised_dataloader, supervised_dataloader, validation_dataloader = dataloaders

        self.pretrain_hidden_layers(max_epochs, unsupervised_dataloader)

        classifier_epochs, classifier_train_losses, classifier_validation_accs = \
            self.train_classifier(max_epochs, supervised_dataloader, validation_dataloader, comparison)

        return classifier_epochs, classifier_train_losses, classifier_validation_accs

    def test_model(self, test_dataloader):
        return accuracy(self.SDAEClassifier, test_dataloader, self.device)

    def classify(self, data):
        self.SDAEClassifier.eval()

        return self.forward(data)

    def forward(self, data):
        return self.SDAEClassifier(data)


def hyperparameter_loop(fold, state_path, results_path, dataset_name, dataloaders, input_size, num_classes, max_epochs,
                        device):
    hidden_layer_size = min(500, (input_size + num_classes) // 2)
    hidden_layers = range(1, 5)
    unsupervised, supervised, validation, test = dataloaders
    train_dataloaders = (unsupervised, supervised, validation)
    num_labelled = len(supervised.dataset)
    lr = 1e-3

    best_acc = 0
    best_params = None

    logging_list = []
    hyperparameter_file = '{}/{}_{}_hyperparameters.p'.format(results_path, fold, num_labelled)
    pickle.dump(logging_list, open(hyperparameter_file, 'wb'))

    for h in hidden_layers:
        print('SDAE hidden layers {}'.format(h))
        logging_list = pickle.load(open(hyperparameter_file, 'rb'))

        model_name = '{}_{}_{}'.format(fold, num_labelled, h)
        model = SDAE(input_size, [hidden_layer_size] * h, num_classes, lr, dataset_name, device, model_name, state_path)
        epochs, losses, val_accs = model.train_model(max_epochs, train_dataloaders, False)
        validation_result = model.test_model(validation)

        model_path = '{}/{}.pt'.format(state_path, model_name)
        torch.save(model.state_dict(), model_path)

        params = {'model name': model_name, 'input size': input_size, 'hidden layers': h * [hidden_layer_size],
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
    model = SDAE(input_size, best_params['hidden layers'], num_classes, lr, dataset_name, device, model_name, state_path)
    model.load_state_dict(torch.load('{}/{}.pt'.format(state_path, model_name)))
    test_acc = model.test_model(test)

    return model_name, test_acc
