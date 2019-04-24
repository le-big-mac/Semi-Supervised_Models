from utils.datautils import *
from torch.utils.data import DataLoader
from Models import *
import argparse
import pickle

parser = argparse.ArgumentParser(description='Take arguments to construct model')
parser.add_argument('imputation_type', type=str, choices=[i.name.lower() for i in ImputationType])
args = parser.parse_args()

imputation_type = ImputationType[args.imputation_type.upper()]
num_labelled = 100000
num_folds = 5
model_name = 'simple_{}'.format(str(imputation_type))
dataset_name = 'tcga_imputations'
max_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_path = './Models/state'
results_path = './results'

if not os.path.exists(state_path):
    os.mkdir(state_path)
if not os.path.exists(results_path):
    os.mkdir(results_path)
if not os.path.exists('{}/{}'.format(results_path, dataset_name)):
    os.mkdir('{}/{}'.format(results_path, dataset_name))
if not os.path.exists('{}/{}'.format(state_path, model_name)):
    os.mkdir('{}/{}'.format(state_path, model_name))
if not os.path.exists('{}/{}/{}'.format(results_path, dataset_name, model_name)):
    os.mkdir('{}/{}/{}'.format(results_path, dataset_name, model_name))
    # clear files
open('./results/{}/{}_logging.p'.format(dataset_name, model_name), 'ab').close()

print('===Loading Data===')
(data, labels), (input_size, num_classes) = load_tcga_data()
str_drop = 'drop_samples' if imputation_type == ImputationType.DROP_SAMPLES else 'nodrop'
folds, _, val_test_split = pickle.load(open('./data/tcga/{}_labelled_{}_folds_{}.p'.format(num_labelled, num_folds,
                                                                                           str_drop)))

results_list = []
pickle.dump(results_list, open('./results/{}/{}_test_results.p'.format(dataset_name, model_name), 'wb'))

for i, (train_indices, val_test_indices) in enumerate(folds):
    results_list = pickle.load(open('./results/mnist/{}_{}_test_results.p'.format(model_name, num_labelled), 'rb'))

    print('Validation Fold {}'.format(i))
    train_data = data[train_indices]
    train_labels = labels[train_indices]

    s_d = TensorDataset(train_data, train_labels)
    u_d = None
    val_test_data = data[val_test_indices]
    val_test_labels = labels[val_test_indices]
    val_indices, test_indices = val_test_split[i]
    v_d = TensorDataset(val_test_data[val_indices], val_test_labels[val_indices])
    t_d = TensorDataset(val_test_data[test_indices], val_test_labels[test_indices])

    u_dl = DataLoader(u_d, batch_size=100, shuffle=True)
    s_dl = DataLoader(s_d, batch_size=100, shuffle=True)
    v_dl = DataLoader(v_d, batch_size=v_d.__len__())
    t_dl = DataLoader(t_d, batch_size=t_d.__len__())

    dataloaders = (u_dl, s_dl, v_dl, t_dl)

    model = SimpleNetwork(input_size, [500, 500], num_classes, 1e-3, dataset_name, device, model_name)
    epochs, losses, validation_accs = model.train_model(max_epochs, dataloaders, False)
    result = model.test_model(t_dl)

    logging = {'test accuracy': result, 'epochs': epochs, 'losses': losses, 'accuracies': validation_accs}

    results_list.append(result)

    print('===Saving Results===')
    pickle.dump(logging, open('./results/{}/{}_logging.p'.format(dataset_name, model_name), 'ab'))
    with open('./results/{}/{}_test_results.p'.format(dataset_name, model_name), 'wb') as test_file:
        pickle.dump(results_list, test_file)
