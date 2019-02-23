import os
import csv


# at some point probably pickle the results and replace list with dictionary (number of labels)
def save_results(results_list, model_file):
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('../results/{}'.format(model_file)):
        os.mkdir('results/{}'.format(model_file))

    accuracy_file = open('results/{}/accuracy.csv'.format(model_file), 'w')
    accuracy_writer = csv.writer(accuracy_file)

    for results in results_list:
        accuracy_writer.writerow(results)
