from utils.datautils import *
from torch.utils.data import DataLoader
from Models import *
import argparse
from collections import defaultdict
import pickle

parser = argparse.ArgumentParser(description='Take arguments to construct model')
parser.add_argument('num_labelled', type=int, help='Number of labelled examples to use')
args = parser.parse_args()

dataset_name = 'mnist'
num_labelled = args.num_labelled
max_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_path = './Models/state'
results_path = './results'
model_folders = ['simple', 'm1', 'sdae', 'm2', 'ladder']

if not os.path.exists(state_path):
    os.mkdir(state_path)
if not os.path.exists(results_path):
    os.mkdir(results_path)
if not os.path.exists('{}/{}'.format(results_path, dataset_name)):
    os.mkdir('{}/{}'.format(results_path, dataset_name))

for name in model_folders:
    if not os.path.exists('{}/{}'.format(state_path, name)):
        os.mkdir('{}/{}'.format(state_path, name))
    if not os.path.exists('{}/{}/{}'.format(results_path, dataset_name, name)):
        os.mkdir('{}/{}/{}'.format(results_path, dataset_name, name))
    # clear files
    open('./results/{}/{}_{}_labelled_hyperparameter_train.csv'.format(dataset_name, name, num_labelled), 'wb').close()

print('===Loading Data===')
(train_data, train_labels), (test_data, test_labels) = load_tcga_data()
t_d = TensorDataset(test_data, test_labels)

results_dict = defaultdict(list)

for i, (train_indices, val_indices) in enumerate(stratified_k_fold(train_data, train_labels, num_folds=5)):
    print('Validation Fold {}'.format(i))

    s_d, u_d = labelled_split(train_data[train_indices], train_labels[train_indices], num_labelled=num_labelled)
    v_d = TensorDataset(train_data[val_indices], train_labels[val_indices])

    u_dl = DataLoader(u_d, batch_size=100, shuffle=True)
    s_dl = DataLoader(s_d, batch_size=100, shuffle=True)
    v_dl = DataLoader(v_d, batch_size=v_d.__len__())
    t_dl = DataLoader(t_d, batch_size=t_d.__len__())

    dataloaders = (u_dl, s_dl, v_dl, t_dl)

    simple_result = simple_hyperparameter_loop(dataset_name, dataloaders, 784, 10, max_epochs, device)
    # m1_result =
    sdae_result = sdae_hyperparameter_loop(dataset_name, dataloaders, 784, 10, max_epochs, device)
    m2_result = m2_hyperparameter_loop(dataset_name, dataloaders, 784, 10, max_epochs, device)

    ladder = LadderNetwork(784, [1000, 500, 250, 250, 250], 10, [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10],
                           1e-3, dataset_name, device)
    ladder_epochs, ladder_loss, ladder_acc = ladder.train_model(100, (u_dl, s_dl, v_dl), False)
    logging = {'epochs': ladder_epochs, 'losses': ladder_loss, 'accuracies': ladder_acc}
    pickle = pickle.dump(logging, open('./results/{}/simple_{}_labelled_hyperparameter_train.csv'
                                       .format(dataset_name, num_labelled)))
    ladder_result = ladder.test_model(t_dl)

    results_dict['simple'].append(simple_result)
    # results_dict['m1'].append(m1_result)
    results_dict['sdae'].append(sdae_result)
    results_dict['m2'].append(m2_result)
    results_dict['ladder'].append(ladder_result)

print('===Saving Results===')
with open('./results/mnist/{}_test_results'.format(num_labelled), 'wb') as test_file:
    pickle.dump(results_dict, test_file)
