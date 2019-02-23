from sklearn.datasets import make_classification
import csv
import os
import random

data, labels = make_classification(n_samples=2000, n_features=500, n_informative=20, n_redundant=20, n_classes=10,
                                   shuffle=False)

# keeps sets balanced
supervised_indices = []
for i in range(0, len(data), 100):
    supervised_indices += list(range(i, i+20))

unsupervised_indices = list(set(range(len(data))) - set(supervised_indices))

supervised_data = [data[i] for i in supervised_indices]
supervised_labels = [labels[i] for i in supervised_indices]

sup = list(zip(supervised_data, supervised_labels))
random.shuffle(sup)

unsupervised_data = [data[i] for i in unsupervised_indices]
random.shuffle(unsupervised_data)

if not os.path.exists('../data'):
    os.mkdir('../data')
    os.mkdir('../data/toy')
elif not os.path.exists('../data/toy'):
    os.mkdir('../data/toy')

unsupervised_data_file = open('../data/toy/toy_unsupervised_data.csv', 'w')
supervised_data_file = open('../data/toy/toy_supervised_data.csv', 'w')
supervised_labels_file = open('../data/toy/toy_supervised_labels.csv', 'w')

unsupervised_data_writer = csv.writer(unsupervised_data_file)
supervised_data_writer = csv.writer(supervised_data_file)
supervised_labels_writer = csv.writer(supervised_labels_file)

for data, label in sup:
    supervised_data_writer.writerow(data)
    supervised_labels_writer.writerow([label])

for data in unsupervised_data:
    unsupervised_data_writer.writerow(data)
