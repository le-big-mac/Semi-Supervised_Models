import sys
import pickle
import argparse
from utils.datautils import *
from Models import *
import torch.nn.functional as F
import csv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Choose mode to run model')
parser.add_argument('mode', type=str, choices=['train', 'classify'], help='Mode to run in')
parser.add_argument('data_filepath', type=str, help='File to load data from')
parser.add_argument('output_folder', type=str, help='Folder to save outputs to')
parser.add_argument('--classification_file', type=str, default='outputs.csv', help='File to save classification '
                                                                                   'results to')
args = parser.parse_args()

mode = args.mode
output_folder = args.output_folder

if not os.path.exists(output_folder):
    print('{} does not exist - making directories'.format(output_folder))
    os.makedirs(output_folder)

state_path = '{}/state'.format(output_folder)

if mode == 'train':
    if not os.path.exists(state_path):
        os.mkdir(state_path)
    else:
        print('WARNING: Possibly overriding previous data in {}'.format(output_folder))
        press = input('Press Enter to continue, or \'e\' followed by Enter to exit:')

        if press == 'e':
            sys.exit()

    print('==Loading Data==')

    (labelled_data, labels), unlabelled_data, label_map, col_means = load_train_data_from_file(args.data_filepath)

    train_val_fold = list(stratified_k_fold(labelled_data, labels, 2))

    print("==M2 optimisation==")

    m2, m2_normalizer, m2_accuracies = m2_tool_loop(train_val_fold, labelled_data, labels, unlabelled_data, output_folder,
                                                    device)

    print("==Ladder optimisation==")

    ladder, ladder_normalizer, ladder_accuracies = ladder_tool_loop(train_val_fold, labelled_data, labels, unlabelled_data,
                                                                    output_folder, device)

    print('==Saving State==')

    torch.save(m2, '{}/m2.pt'.format(state_path))
    pickle.dump(m2_normalizer, open('{}/m2_normalizer.p'.format(state_path), 'wb'))

    torch.save(ladder, '{}/ladder.pt'.format(state_path))
    pickle.dump(ladder_normalizer, open('{}/ladder_normalizer.p'.format(state_path), 'wb'))

    pickle.dump(col_means, open('{}/imputation_means.p'.format(state_path), 'wb'))
    pickle.dump(label_map, open('{}/label_map.p'.format(state_path), 'wb'))

    print('M2 2-fold accuracies: {}'.format(m2_accuracies))
    print('Ladder 2-fold accuracies: {}'.format(ladder_accuracies))

    m2_accuracies.insert(0, 'M2')
    ladder_accuracies.insert(0, "Ladder")

    f = open("{}/accs.csv".format(output_folder), 'w')
    writer = csv.writer(f)
    writer.writerow(m2_accuracies)
    writer.writerow(ladder_accuracies)
    f.close()

if mode == 'classify':
    class_file = args.classification_file

    if class_file == 'outputs.csv':
        print('WARNING: Using default output file outputs.csv. This may override previous data')
        press = input('Press Enter to continue, or \'e\' followed by Enter to exit:')

        if press == 'e':
            sys.exit()

    if not (os.path.exists('{}/m2.pt'.format(state_path)) or os.path.exists('{}/ladder.pt'.format(state_path))):
        sys.exit('Models have not been trained')

    print('==Loading Data==')

    col_means = pickle.load(open('{}/imputation_means.p'.format(state_path), 'rb'))
    int_string_map = pickle.load(open('{}/label_map.p'.format(state_path), 'rb'))
    sample_names, data = load_data_to_classify_from_file(args.data_filepath, col_means)

    if device.type == 'cpu':
        m2 = torch.load('{}/m2.pt'.format(state_path), map_location='cpu')
        ladder = torch.load('{}/ladder.pt'.format(state_path), map_location='cpu')
    else:
        m2 = torch.load('{}/m2.pt'.format(state_path))
        ladder = torch.load('{}/ladder.pt'.format(state_path))

    print('==Classifying==')

    m2_normalizer = pickle.load(open('{}/m2_normalizer.p'.format(state_path), 'rb'))
    ladder_normalizer = pickle.load(open('{}/ladder_normalizer.p'.format(state_path), 'rb'))

    m2_data = torch.tensor(m2_normalizer.transform(data)).float()
    m2_results = m2.classify(data)

    ladder_data = torch.tensor(ladder_normalizer.transform(data)).float()
    ladder_results = ladder.classify(data)

    predictions = (F.softmax(m2_results, dim=1) + F.softmax(ladder_results, dim=1))/2

    _, predictions = torch.max(predictions.data, 1)

    labels = [[sample_names[i], int_string_map[l]] for i, l in enumerate(predictions.cpu().numpy())]

    print('==Writing Output==')

    file = open('{}/{}'.format(output_folder, class_file), 'w')
    writer = csv.writer(file)

    for row in labels:
        writer.writerow(row)

    file.close()

