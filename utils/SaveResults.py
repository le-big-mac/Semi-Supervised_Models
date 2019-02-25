import os
import csv


# at some point probably pickle the results and replace list with dictionary (number of labels)
def save_results(results_list, model_directory, filename):
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/{}'.format(model_directory)):
        os.mkdir('results/{}'.format(model_directory))

    accuracy_file = open('results/{}/{}.csv'.format(model_directory, filename), 'w')
    accuracy_writer = csv.writer(accuracy_file)

    accuracy_writer.writerow(results_list)

    accuracy_file.close()
