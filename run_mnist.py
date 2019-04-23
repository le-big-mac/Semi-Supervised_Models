from utils.datautils import *
from torch.utils.data import DataLoader
from Models import *

model_name = 'm2'
dataset_name = 'mnist'
input_size = 784
num_classes = 10
num_labelled = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_path = './Models/state'
results_path = './results'

if not os.path.exists(state_path):
    os.mkdir(state_path)
if not os.path.exists('{}/{}'.format(state_path, model_name)):
    os.mkdir('{}/{}'.format(state_path, model_name))
if not os.path.exists(results_path):
    os.mkdir(results_path)
if not os.path.exists('{}/{}'.format(results_path, dataset_name)):
    os.mkdir('{}/{}'.format(results_path, dataset_name))
if not os.path.exists('{}/{}/{}'.format(results_path, dataset_name, model_name)):
    os.mkdir('{}/{}/{}'.format(results_path, dataset_name, model_name))

print('===Loading Data===')
(train_data, train_labels), (test_data, test_labels) = load_MNIST_data()
t_d = TensorDataset(test_data, test_labels)

test_accuracies = []
validation_accuracies = []
train_losses = []
train_epochs = []

for i, (train_indices, val_indices) in enumerate(stratified_k_fold(train_data, train_labels, num_folds=5)):
    print('Fold {}'.format(i))

    print('===Making Model===')
    model = M2Runner(input_size, [397, 397], [397], 50, num_classes, lambda x: x, 1e-3, dataset_name, device)

    s_d, u_d = \
        labelled_split(train_data[train_indices], train_labels[train_indices], num_labelled=num_labelled)

    v_d = TensorDataset(train_data[val_indices], train_labels[val_indices])

    u_dl = DataLoader(u_d, batch_size=100, shuffle=True)
    s_dl = DataLoader(s_d, batch_size=100, shuffle=True)
    v_dl = DataLoader(v_d, batch_size=v_d.__len__())
    t_dl = DataLoader(t_d, batch_size=t_d.__len__())

    epochs, losses, accuracies = model.train_model(300, (u_dl, s_dl, v_dl), False)
    test_accuracy = model.test_model(t_dl)

    print('Test accuracy: {}'.format(test_accuracy))

    train_epochs.append(epochs)
    train_losses.append(losses)
    validation_accuracies.append(accuracies)
    test_accuracies.append(test_accuracy)

print('===Saving Results===')
save_results(test_accuracies, dataset_name, model_name,
             '{}_labelled_test_accuracies'.format(num_labelled))
save_results(train_epochs, dataset_name, model_name,
             '{}_labelled_epochs'.format(num_labelled))
save_results(train_losses, dataset_name, model_name,
             '{}_labelled_train_losses'.format(num_labelled))
save_results(validation_accuracies, dataset_name, model_name,
             '{}_labelled_validation_accuracies'.format(num_labelled))
