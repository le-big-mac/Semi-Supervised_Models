from torch import nn


class Model(nn.Module):
    def __init__(self, dataset_name, device):
        super(Model, self).__init__()
        self.dataset_name = dataset_name
        self.device = device

    def train_model(self,  max_epochs, dataloaders, comparison):
        raise NotImplementedError

    def test_model(self, test_dataloader):
        raise NotImplementedError

    def classify(self, data):
        raise NotImplementedError

    def forward(self, data):
        raise NotImplementedError
