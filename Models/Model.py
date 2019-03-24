class Model(object):
    def __init__(self, dataset_name, device):
        self.dataset_name = dataset_name
        self.device = device

    def train(self, dataset_name, supervised_dataloader, unsupervised_dataloader, validation_dataloader=None):
        raise NotImplementedError

    def test(self, test_dataset):
        raise NotImplementedError

    def classify(self, dataset):
        # TODO: implement this so that it saves to a file in the models
        raise NotImplementedError
