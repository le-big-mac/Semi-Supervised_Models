import torch
from torch import nn
from Models.BuildingBlocks import Classifier
from Models.Model import Model
from utils.trainingutils import accuracy, EarlyStopping
import pickle


class SimpleNetwork(Model):
    def __init__(self, input_size, hidden_dimensions, num_classes, lr, dataset_name, device):
        super(SimpleNetwork, self).__init__(dataset_name, device)

        self.Classifier = Classifier(input_size, hidden_dimensions, num_classes).to(device)
        self.optimizer = torch.optim.Adam(self.Classifier.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.model_name = 'simple'

    def train_classifier(self, max_epochs, train_dataloader, validation_dataloader, comparison):
        epochs = []
        train_losses = []
        validation_accs = []

        early_stopping = EarlyStopping('{}/{}.pt'.format(self.model_name, self.dataset_name))

        print(accuracy(self.Classifier, validation_dataloader, self.device))
        for epoch in range(max_epochs):
            if early_stopping.early_stop:
                break

            train_loss = 0
            for batch_idx, (data, labels) in enumerate(train_dataloader):
                self.Classifier.train()

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                preds = self.Classifier(data)

                loss = self.criterion(preds, labels)

                loss.backward()
                self.optimizer.step()

                if comparison:
                    acc = accuracy(self.Classifier, validation_dataloader, self.device)

                    epochs.append(epoch)
                    train_losses.append(loss.item())
                    validation_accs.append(acc)

                    early_stopping(1 - acc, self.Classifier)
                else:
                    train_loss += loss.item()

            if not comparison:
                acc = accuracy(self.Classifier, validation_dataloader, self.device)

                epochs.append(epoch)
                train_losses.append(train_loss/len(train_dataloader))
                validation_accs.append(acc)

                early_stopping(1 - acc, self.Classifier)

        if early_stopping.early_stop:
            early_stopping.load_checkpoint(self.Classifier)

        return epochs, train_losses, validation_accs

    def train_model(self, max_epochs, dataloaders, comparison):
        _, supervised_dataloader, validation_dataloader = dataloaders

        epochs, losses, validation_accs = self.train_classifier(max_epochs, supervised_dataloader,
                                                                validation_dataloader, comparison)

        return epochs, losses, validation_accs

    def test_model(self, test_dataloader):
        return accuracy(self.Classifier, test_dataloader, self.device)

    def classify(self, data):
        self.Classifier.eval()

        return self.forward(data)

    def forward(self, data):
        return self.Classifier(data.to(self.device))


def hyperparameter_loop(dataset_name, dataloaders, input_size, num_classes, max_epochs, device):
    hidden_layer_size = min(500, (input_size + num_classes) // 2)
    hidden_layers = range(1, 5)
    unsupervised, supervised, validation, test = dataloaders
    train_dataloaders = (unsupervised, supervised, validation)
    num_labelled = len(supervised.dataset)
    lr = 1e-3

    best_acc = 0
    best_path = None
    best_params = None

    f = open('./results/{}/simple_{}_labelled_hyperparameter_train.csv'.format(dataset_name, num_labelled), 'ab')
    for h in hidden_layers:
        print('Simple hidden layers {}'.format(h))

        model = SimpleNetwork(input_size, [hidden_layer_size] * h, num_classes, lr, dataset_name, device)
        epochs, losses, val_accs = model.train_model(max_epochs, train_dataloaders, False)
        validation_result = model.test_model(validation)

        model_path = './state/simple/{}_{}_{}'.format(dataset_name, num_labelled, h)
        torch.save(model.state_dict(), model_path)

        params = {'input size': input_size, 'hidden layers': h * [hidden_layer_size], 'num classes': num_classes}
        logging = {'accuracy': validation_result, 'epochs': epochs, 'losses': losses, 'accuracies': validation_result,
                   'params': params, 'filepath': model_path}

        pickle.dump(logging, f)

        if validation_result > best_acc:
            best_acc = validation_result
            best_path = model_path
            best_params = params

        if device == 'cuda':
            torch.cuda.empty_cache()

    f.close()

    model = SimpleNetwork(input_size, best_params['hidden layers'], num_classes, lr, dataset_name, device)
    model.load_state_dict(torch.load(best_path))
    test_acc = model.test_model(test)

    return test_acc
