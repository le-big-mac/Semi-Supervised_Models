import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


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


class MNISTSupervised(Dataset):
    def __init__(self, data, labels):

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        # Thanks Yann LeCun
        img = transforms.ToTensor()(Image.fromarray(img.numpy(), mode='L')).view(784,)

        return img, label


class MNISTUnsupervised(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = transforms.ToTensor()(Image.fromarray(img.numpy(), mode='L')).view(784,)
        return img
