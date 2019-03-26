import torch


def accuracy(model, dataloader, device):
    model.eval()

    correct = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            outputs = model(data.float().to(device))
            labels = labels.to(device)
 
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return correct / len(dataloader.dataset)


def unsupervised_validation_loss(model, dataloader, criterion, device):
    model.eval()

    validation_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)

        recons = model(data)

        loss = criterion(recons, data)

        validation_loss += loss.item()

    return validation_loss/len(dataloader)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_filename, patience=20, delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.filename = './Models/state/{}'.format(checkpoint_filename)
        self.patience = patience
        self.delta = abs(delta)
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                  .format(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), self.filename)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename))
