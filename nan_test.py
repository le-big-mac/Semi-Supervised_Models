from utils.datautils import *
import pickle

(data, labels), (input_size, num_classes) = load_tcga_data(ImputationType.DROP_SAMPLES)
folds, labelled_indices, val_test_split = pickle.load(open('./data/tcga/10000_labelled_5_folds_drop_samples.p'))

for i, (train_indices, test_val_indices) in enumerate(folds):
    normalizer = GaussianNormalizeTensors()

    train_data = normalizer.apply_train(data[train_indices])
    test_val_data = normalizer.apply_test(data[test_val_indices])

    print(torch.sum(torch.isnan(train_data)).item() > 0)
    print(torch.sum(torch.isnan(test_val_data)).item() > 0)
