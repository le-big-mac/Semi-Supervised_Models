from copy import deepcopy


class Saliency(object):

    def __init__(self, model, device):
        self.model = deepcopy(model)
        self.model.eval()
        self.device = device

    def generate_saliency(self, input, target):
        raise NotImplementedError
