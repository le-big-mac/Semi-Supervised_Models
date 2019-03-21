import argparse
from Models import SimpleNetwork, PretrainingNetwork, SDAENet

parser = argparse.ArgumentParser(description="Take arguments to construct model")
parser.add_argument("model", type=str,
                    choices=['simple', 'pretraining', 'sdae', 'm1', 'm2', 'ladder'],
                    help="Choose which model to run"
                    )
# parser.add_argument("unsupervised_file", type=str,
#                         help="Relative path to file containing data for unsupervised training")
# parser.add_argument("supervised_data_file", type=str,
#                         help="Relative path to file containing the input data for supervised training")
# parser.add_argument("supervised_labels_file", type=str,
#                         help="Relative path to file containing the output data for supervised training")

args = parser.parse_args()

model = None
model_name = args.model

if model_name == 'simple':
    model =

