import torch
from Saliency.saliency import Saliency


class VanillaSaliency(Saliency):
    """Vanilla Saliency to visualize plain gradient information"""

    def __init__(self, model, device):
        super(VanillaSaliency, self).__init__(model, device)

    def generate_saliency(self, input, target):
        input.requires_grad = True

        self.model.zero_grad()

        output = self.model(input.to(self.device))

        grad_outputs = torch.zeros_like(output)

        grad_outputs[:, target] = 1

        output.backward(gradient=grad_outputs)

        input.requires_grad = False

        return input.grad.clone()[0]
