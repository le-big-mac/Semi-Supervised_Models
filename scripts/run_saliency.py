import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.datautils import load_MNIST_data, stratified_k_fold
from Saliency import VanillaSaliency, GuidedSaliency
import matplotlib.pyplot as plt
from Models import SimpleNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

saliency_map = {
    'vanilla': VanillaSaliency,
    'guided': GuidedSaliency
}


def __main__():
    (data, labels), (t, l) = load_MNIST_data()

    model = SimpleNetwork(784, [400, 400], 10, 1e-3, 'saliency', device, 'simple_saliency', './outputs/saliency')

    indices = stratified_k_fold(data, labels, 2)
    train, val = next(indices)

    train_dataset = TensorDataset(data[train], labels[train])
    val_dataset = TensorDataset(data[val], labels[val])
    t_dl = DataLoader(train_dataset, batch_size=100, shuffle=True)
    v_dl = DataLoader(val_dataset, batch_size=val_dataset.__len__())

    model.train_model(100, (None, t_dl, v_dl))

    input = t[12].unsqueeze(0)
    output = model.classify(input)

    _, prediction = output.max(1)

    print(prediction)

    vanilla_saliency = VanillaSaliency(model.Classifier, device).generate_saliency(input, prediction)
    guided_saliency = GuidedSaliency(model.Classifier, device).generate_saliency(input, prediction)

    if device.type == 'cuda':
        vanilla_saliency = vanilla_saliency.cpu()
        guided_saliency = guided_saliency.cpu()

    for s in [(vanilla_saliency, 'vanilla'), (guided_saliency, 'guided')]:
        saliency, string = s

        pos_map = saliency.clamp(min=0)
        pos_map = pos_map / pos_map.max()

        neg_map = - saliency.clamp(max=0)
        neg_map = neg_map / neg_map.max()

        abs_map = saliency.abs()
        abs_map = abs_map / abs_map.max()

        input = input.detach()
        input = input.view(28, 28)
        pos_map = pos_map.view(28, 28)
        neg_map = neg_map.view(28, 28)
        abs_map = abs_map.view(28, 28)

        figure = plt.figure(figsize=(8, 8), facecolor='w')

        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(input, cmap="gray", interpolation=None)

        plt.subplot(2, 2, 2)
        plt.title("Positive Saliency")
        plt.imshow(pos_map, cmap='gray', interpolation=None)

        plt.subplot(2, 2, 3)
        plt.title("Negative Saliency")
        plt.imshow(neg_map, cmap='gray', interpolation=None)

        plt.subplot(2, 2, 4)
        plt.title("Absolute Saliency")
        plt.imshow(abs_map, cmap='gray', interpolation=None)

        plt.savefig('./outputs/saliency/{}_saliency_maps.pdf'.format(string))


if __name__ == "__main__":
    __main__()
