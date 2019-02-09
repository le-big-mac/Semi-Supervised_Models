import torch
import numpy as np
from torch.utils.data import Dataset


class SupervisedDataset(Dataset):

    def __init__(self, inputs, outputs):
        super(SupervisedDataset, self).__init__()

        self.raw_input = inputs
        self.raw_output = outputs
        self.inputs = [torch.from_numpy(np.atleast_1d(vec)) for vec in inputs]
        self.outputs = [torch.from_numpy(np.atleast_1d(vec)) for vec in outputs]

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]


class UnsupervisedDataset(Dataset):

    def __init__(self, inputs):
        super(UnsupervisedDataset, self).__init__()

        self.raw_data = inputs
        self.data = [torch.from_numpy(np.atleast_1d(vec)) for vec in inputs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
