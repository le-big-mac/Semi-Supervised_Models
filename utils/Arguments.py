import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Take arguments to construct model")
    parser.add_argument("unsupervised_file", type=str,
                        help="Relative path to file containing data for unsupervised training")
    parser.add_argument("supervised_input_file", type=str,
                        help="Relative path to file containing the input data for supervised training")
    parser.add_argument("output_file", type=str,
                        help="Relative path to file containing the output data for supervised training")
    parser.add_argument("--model", dest="unsupervised_model_file", type=str, default="model.pt",
                        help="Relative path to file to store the trained unsupervised model")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=500, help="Batch size to use in training")
    parser.add_argument("--epochs_unsupervised", dest="num_epochs_unsupervised", type=int, default=100,
                        help="Number of epochs to train for unsupervised")
    parser.add_argument("--epochs_supervised", dest="num_epochs_supervised", type=int, default=100,
                        help="Number of epochs to train for supervised")
    parser.add_argument("--num_folds", dest="num_folds", type=int, default=10,
                        help="Number of folds for cross validation")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-3,
                        help="Learning rate for gradient descent")
    args = parser.parse_args()

    return args
