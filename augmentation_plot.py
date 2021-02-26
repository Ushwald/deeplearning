import matplotlib.pyplot as plt
import pandas as pd
import logger
from argument_parser import setup_argument_parser
import os
import seaborn as sns
from matplotlib.lines import Line2D

log = logger.setup_logger(__name__)

def main():
    config = setup_argument_parser()
    if config.crossvalidation:
        log.info("crossval not implemented!")
    else:
        augmentation_plot(config)
    
    

def augmentation_plot(config):

    palette = sns.color_palette(n_colors=2)
    # data

    fp1 = f"model_training_history/{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}\
Optimizer-TrueAugmentation_lrate{config.learningrate}_mom{config.momentum}-history.csv"
    fp2 = f"model_training_history/{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}\
Optimizer-FalseAugmentation_lrate{config.learningrate}_mom{config.momentum}-history.csv"



    try:
        aug_data = pd.read_csv(fp1, sep = ',')
    except:
        log.error(f"history csv files were not successfully opened (filepath: {fp1})")

    try:
        no_aug_data = pd.read_csv(fp2, sep = ',')
    except:
        log.error(f"history csv files were not successfully opened (filepath: {fp2})")


    epochs_range = range(len(aug_data.index))
    aug_accuracy = aug_data["accuracy"]
    aug_val_accuracy = aug_data["val_accuracy"]

    # create plot
    fig, ax = plt.subplots()

    # apply datak
    aug_accuracy = aug_data["accuracy"]
    aug_val_accuracy = aug_data["val_accuracy"]
    no_aug_accuracy = no_aug_data["accuracy"]
    no_aug_val_accuracy = no_aug_data["val_accuracy"]

    epochs_range = range(len(aug_data.index))
    ax.plot(epochs_range, aug_accuracy, label="training accuracy (Augmentation)", linestyle="dotted", color = palette[0])
    ax.plot(epochs_range, aug_val_accuracy, label="validation accuracy (Augmentation)", linestyle="dashed", color = palette[0])
    
    ax.plot(epochs_range, no_aug_accuracy, label="training accuracy (No augmentation)", linestyle="dotted", color = palette[1])
    ax.plot(epochs_range, no_aug_val_accuracy, label="validation accuracy (No augmentation)", linestyle="dashed", color = palette[1])


    # draw boilerplate
    ax.set(
        xlabel="epoch", ylabel="Accuracy", title="Training and Validation Accuracy"
    )
    if config.optimizer == 'rms':
        optimstr = "RMSprop"
    else:
        optimstr = "SGD"

    caption = f"Accuracies with settings: Activation: {config.activation}, Optimizer: {optimstr}, \n\
Learning rate: {config.learningrate}, Momentum: {config.momentum}"
    fig.text(0.5, 0.02, caption, ha="center", style="italic")

    plt.legend(loc="upper left")

    # stylize
    ax.grid()
    fig.subplots_adjust(bottom=0.2)
    # show and export
    try:
        os.mkdir("./results/")
    except:
        log.info('results folder already existed')
    fig.savefig(f"./results/{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}\
    Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}_augmentationplot.jpg")

if __name__ == "__main__":
    main()
