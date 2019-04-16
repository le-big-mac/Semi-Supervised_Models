import torch
import argparse
from utils.datautils import load_data_from_file
from Saliency import VanillaSaliency, GuidedSaliency
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_path = './Models/state'

saliency_map = {
    'vanilla': VanillaSaliency,
    'guided':GuidedSaliency
}


def __main__():
    parser = argparse.ArgumentParser(description='Data and model to compute saliency for')
    parser.add_argument('model_type', type=str,
                        choices=['simple', 'pretraining', 'sdae', 'simple_m1', 'm1', 'm2', 'ladder'],
                        help="Choose which model to run"
                        )
    parser.add_argument('model_name')
    parser.add_argument('data')
    parser.add_argument('--saliency_type')

    args = parser.parse_args()
    model_type = args.model_type

    data = load_data_from_file(args.data)

    # TODO: no good way of loading models up

    input = test_dataset[input_idx][0].unsqueeze(0)

    output = model.classify(input)

    _, prediction = output.max(1)

    print(prediction)

    vanilla_saliency = VanillaSaliency(model.Classifier, device).generate_saliency(input, prediction)
    guided_saliency = GuidedSaliency(model.Classifier, device).generate_saliency(input, prediction)

    if device.type == 'cuda':
        guided_saliency = guided_saliency.cpu()

    pos_map = guided_saliency.clamp(min=0)
    pos_map = pos_map / pos_map.max()

    neg_map = - guided_saliency.clamp(max=0)
    neg_map = neg_map / neg_map.max()

    abs_map = guided_saliency.abs()
    abs_map = abs_map / abs_map.max()

    input = input.detach()
    input = input.view(28, 28)
    pos_map = pos_map.view(28, 28)
    neg_map = neg_map.view(28, 28)
    abs_map = abs_map.view(28, 28)
    #
    # plt.imsave('original.png', input, cmap='gray')
    # plt.imsave('pos.png', pos_map, cmap='gray')
    # plt.imsave('neg.png', neg_map, cmap='gray')
    # plt.imsave('abs.png', abs_map, cmap='gray')

    figure = plt.figure(figsize=(8, 8), facecolor='w')

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(input, cmap="gray")

    plt.subplot(2, 2, 2)
    plt.title("Positive Saliency")
    plt.imshow(pos_map, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title("Negative Saliency")
    plt.imshow(neg_map, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title("Absolute Saliency")
    plt.imshow(abs_map, cmap='gray')

    plt.show()