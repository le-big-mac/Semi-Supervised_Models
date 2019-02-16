import torch
import numpy as np
from torch.utils.data import Dataset


class SupervisedClassificationDataset(Dataset):

    def __init__(self, inputs, outputs):
        super(SupervisedClassificationDataset, self).__init__()

        self.inputs = [torch.from_numpy(np.atleast_1d(vec)).float() for vec in inputs]
        self.outputs = [torch.squeeze(torch.from_numpy(np.atleast_1d(vec)).long()) for vec in outputs]

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]


class UnsupervisedDataset(Dataset):

    def __init__(self, inputs):
        super(UnsupervisedDataset, self).__init__()

        self.data = [torch.from_numpy(np.atleast_1d(vec)).float() for vec in inputs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
