from copy import deepcopy


class Saliency(object):

    def __init__(self, model):
        self.model = deepcopy(model)
        self.model.eval()

    def generate_saliency(self, input, target):
        raise NotImplementedError
