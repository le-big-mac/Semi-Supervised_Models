import torch
from torch import nn
import torch.nn.functional as F
from utils.datautils import load_MNIST_data
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        layers = [784, 1000, 500, 250, 250, 250]

        self.h = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.out = nn.Linear(250, 10)

    def forward(self, x):
        for h in self.h:
            x = F.relu(h(x))

        return self.out(x)


_, train_dataset, validation_dataset, _ = load_MNIST_data(100, 10, True, True)
train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=validation_dataset.__len__())


def accuracy(model):
    model.eval()

    correct = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(validation_dataloader):
            outputs = model(data.float().to(device))
            labels = labels.to(device)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return correct / len(validation_dataloader.dataset)


for i in range(5):
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        for data, labels in train_dataloader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            pred = model(data)

            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()

        val = accuracy(model)

        print('Epoch: {} Validation Acc: {}'.format(epoch, val))
