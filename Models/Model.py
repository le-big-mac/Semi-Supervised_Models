class Model(object):
    def __init__(self, device):
        self.device = device

    def train(self, *datasets):
        raise NotImplementedError

    def test(self, test_dataset):
        raise NotImplementedError

    def classify(self, dataset):
        # TODO: implement this so that it saves to a file in the models
        raise NotImplementedError
