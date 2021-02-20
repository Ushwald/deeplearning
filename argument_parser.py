import argparse

def setup_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help = "Number of epochs to train for", type = int, default = 10)
    parser.add_argument("--activation", help = "Hidden layer activation function", choices = ['relu', 'other'], type = str, default = 'relu')
    parser.add_argument("--optimizer", help = "Specified optimizer algorithm", choices = ['rms', 'sgdm'], type=str, default = 'rms')
    parser.add_argument("--augmentation", help = "Boolean: whether augmentation should be applied to training data", type = bool, default = False)

    return parser.parse_args()