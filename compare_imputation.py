from utils.datautils import *
from torch.utils.data import DataLoader
from Models import *
import argparse
import pickle

parser = argparse.ArgumentParser(description='Take arguments to construct model')
parser.add_argument('imputation_type', type=str, choices=[i.name.lower() for i in ImputationType])
args = parser.parse_args()

imputation_string = args.imputation_type.upper()
imputation_type = ImputationType[imputation_string]
print(ImputationType)
num_labelled = 100000
num_folds = 5
model_name = 'simple_{}'.format(str(imputation_string))
dataset_name = 'tcga_imputations'
max_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_path = './outputs'
results_path = '{}/{}/{}/results'.format(output_path, dataset_name, model_name)
state_path = '{}/{}/{}/state'.format(output_path, dataset_name, model_name)

if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists('{}/{}'.format(output_path, dataset_name)):
    os.mkdir('{}/{}'.format(output_path, dataset_name))
if not os.path.exists('{}/{}/{}'.format(output_path, dataset_name, model_name)):
    os.mkdir('{}/{}/{}'.format(output_path, dataset_name, model_name))
if not os.path.exists(results_path):
    os.mkdir(results_path)
if not os.path.exists(state_path):
    os.mkdir(state_path)

print('===Loading Data===')
(data, labels), (input_size, num_classes) = load_tcga_data(imputation_type)
str_drop = 'drop_samples' if imputation_type == ImputationType.DROP_SAMPLES else 'no_drop'
folds, _, val_test_split = pickle.load(open('./data/tcga/{}_labelled_{}_folds_{}.p'.format(num_labelled, num_folds,
                                                                                           str_drop), 'rb'))

results_list = []
pickle.dump(results_list, open('{}/test_results.p'.format(results_path), 'wb'))

for i, (train_indices, val_test_indices) in enumerate(folds):
    results_list = pickle.load(open('{}/test_results.p'.format(results_path), 'rb'))

    print('Validation Fold {}'.format(i))
    normalizer = GaussianNormalizeTensors()
    train_data = normalizer.apply_train(data[train_indices])
    train_labels = labels[train_indices]

    print('Train size: {}'.format(len(train_indices)))

    s_d = TensorDataset(train_data, train_labels)
    u_d = None
    val_test_data = normalizer.apply_test(data[val_test_indices])
    val_test_labels = labels[val_test_indices]
    val_indices, test_indices = val_test_split[i]

    print('Val size: {}'.format(len(val_indices)))
    print('Test size: {}'.format(len(test_indices)))

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
    pickle.dump(logging, open('{}/{}_logging.p'.format(results_path, i), 'wb'))
    pickle.dump(results_list, open('{}/test_results.p'.format(results_path), 'wb'))
