import torch
import torch.nn as nn
from Saliency.saliency import Saliency


class GuidedSaliency(Saliency):
    """Class for computing guided saliency"""
    def __init__(self, model, device):
        super(GuidedSaliency, self).__init__(model, device)
        self.forward_relu_outputs = []

    def relu_backward_hook_function(self, module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero
        """
        # Get last forward output
        corresponding_forward_output = self.forward_relu_outputs[-1]
        positive_mask = (corresponding_forward_output > 0).type_as(corresponding_forward_output)

        # add zeros to avoid -0.0
        modified_grad_out = positive_mask * torch.clamp(grad_in[0], min=0.0)

        del self.forward_relu_outputs[-1]  # Remove last forward output
        return (modified_grad_out,)

    def relu_forward_hook_function(self, module, ten_in, ten_out):
        """
        Store results of forward pass
        """
        self.forward_relu_outputs.append(ten_out)

    def generate_saliency(self, input, target):
        input.requires_grad = True

        self.model.zero_grad()

        for module in self.model.modules():
            if type(module) == nn.ReLU:
                module.register_forward_hook(self.relu_forward_hook_function)
                module.register_backward_hook(self.relu_backward_hook_function)

        output = self.model(input.to(self.device))

        grad_outputs = torch.zeros_like(output)

        grad_outputs[:, target] = 1

        self.model.zero_grad()

        output.backward(gradient=grad_outputs)

        return input.grad.clone()[0]
