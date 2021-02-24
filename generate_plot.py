import matplotlib.pyplot as plt
import pandas as pd
from argument_parser import setup_argument_parser
import os
import seaborn as sns

log = logger.setup_logger(__name__)

def main():
    config = setup_argument_parser()
    if config.crossvalidation:
        plot_crossvalidation(config)
    else:
        plot(config)
    
    print(data.head())
    plot(data)

def plot_crossvalidation(config):
    fig, ax = plt.subplots()
    palette = sns.color_palette(5)

    for kfold in range(5):
            # Plot lines for training and validation accuracies for all folds:
            fp = f"model_training_history/Foldnr{kfold+1}_{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}\
            Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}-history.csv"
            try:
                data = pd.read_csv(fp, sep = ',')
            except:
                log.error("kfold crossvalidation history csv files were not successfully opened")

            accuracy = data["accuracy"]
            val_accuracy = data["val_accuracy"]
            epochs_range = range(len(data.index))
            ax.plot(epochs_range, loss, label="training loss", linestyle="dotted",color = palette[kfold])
            ax.plot(epochs_range, val_accuracy, label="validation loss", linestyle="dashed", color = palette[kfold])

    ax.set(
        xlabel="epoch", ylabel="Accuracy", title="Training and Validation Accuracy"
    )
    if config.optimizer == 'rms':
        optimstr = "RMSprop"
    else:
        optimstr = "SGD"
        
    caption(f"5-fold crossvalidation with settings: Activation: {config.activation}, Optimizer: {optimstr}, \
Learning rate: {config.learningrate}, Momentum: {config.momentum}")
    fig.text(0.5, 0.02, caption, ha="center", style="italic")
    plt.legend(loc="upper right")
    ax.grid()
    fig.subplots_adjust(bottom=0.2)

    fig.savefig(f"./crossvalidation_results/{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}\
    Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}.jpg")


def plot(data):
    # data
    fp = f"model_training_history/Foldnr{kfold+1}_{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}\
Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}-history.csv"

    try:
        data = pd.read_csv(fp, sep = ',')
    except:
        log.error("kfold crossvalidation history csv files were not successfully opened")

    epochs_range = range(len(data.index))
    loss = data["loss"]
    val_loss = data["val_loss"]

    # create plot
    fig, ax = plt.subplots()

    # apply data
    accuracy = data["accuracy"]
    val_accuracy = data["val_accuracy"]
    epochs_range = range(len(data.index))
    ax.plot(epochs_range, loss, label="training loss", linestyle="dotted")
    ax.plot(epochs_range, val_accuracy, label="validation loss", linestyle="dashed")

    # draw boilerplate
    ax.set(
        xlabel="epoch", ylabel="Accuracy", title="Training and Validation Accuracy"
    )
    if config.optimizer == 'rms':
        optimstr = "RMSprop"
    else:
        optimstr = "SGD"

    caption(f"Accuracies with settings: Activation: {config.activation}, Optimizer: {optimstr}, \
Learning rate: {config.learningrate}, Momentum: {config.momentum}, Augmentation = {config.augmentation}")
    fig.text(0.5, 0.02, caption, ha="center", style="italic")

    plt.legend(loc="upper right")

    # stylize
    ax.grid()
    fig.subplots_adjust(bottom=0.2)
    # show and export
    fig.savefig(f"./results/{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}\
    Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}.jpg")


if __name__ == "__main__":
    main()
