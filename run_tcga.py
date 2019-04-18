from utils.datautils import *
from torch.utils.data import DataLoader
from Models import *

model_name = 'm2'
dataset_name = 'tcga'
num_labelled = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('===Loading Data===')
(data, labels), input_size, num_classes = load_tcga_data()

test_accuracies = []
validation_accuracies = []
train_losses = []
train_epochs = []

for i, train_indices, val_and_test_indices in enumerate(stratified_k_fold(data, labels, num_folds=5)):
    print('Fold {}'.format(i))

    print('===Making Model===')
    model = M2Runner(input_size, [1024, 1024], [1024], 32, num_classes, lambda x: x, 1e-3, dataset_name, device)

    normalizer = GaussianNormalizeTensors()
    train_data = normalizer.apply_train(data[train_indices])

    s_d, u_d = \
        labelled_split(train_data, labels[train_indices], num_labelled=num_labelled)

    val_and_test_data = normalizer.apply_test(data[val_and_test_indices])
    val_and_test_labels = labels[val_and_test_indices]
    val_indices, test_indices = next(stratified_k_fold(val_and_test_data, val_and_test_labels,
                                                       num_folds=2))
    v_d = TensorDataset(val_and_test_data[val_indices], val_and_test_labels[val_indices])
    t_d = TensorDataset(val_and_test_data[test_indices], val_and_test_labels[test_indices])

    u_dl = DataLoader(u_d, batch_size=100, shuffle=True)
    s_dl = DataLoader(s_d, batch_size=100, shuffle=True)
    v_dl = DataLoader(v_d, batch_size=v_d.__len__())
    t_dl = DataLoader(t_d, batch_size=t_d.__len__())

    epochs, losses, accuracies = model.train_model(300, (u_dl, s_dl, v_dl), True)
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
