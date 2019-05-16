from torch import nn


class Model(nn.Module):
    def __init__(self, device, state_path, model_name):
        super(Model, self).__init__()
        self.device = device
        self.state_path = state_path
        self.model_name = model_name

    def train_model(self,  max_epochs, dataloaders):
        raise NotImplementedError

    def test_model(self, test_dataloader):
        raise NotImplementedError

    def classify(self, data):
        raise NotImplementedError

    def forward(self, data):
        raise NotImplementedError
