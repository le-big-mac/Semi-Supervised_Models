import torch


def accuracy(model, dataloader):
    model.eval()

    correct = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return correct / len(dataloader.dataset)
