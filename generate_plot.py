import matplotlib.pyplot as plt
import pandas as pd
import os

name = "./model_training_history/2Epochs_reluActivation-rmsOptimizer-FalseAugmentation-history.csv"
p = {"dor": 0.25, "ilr": 0.001, "opt": "adamax", "pat": 15, "bts": 16}


def main():
    data = pd.read_csv(name, sep=",")
    print(data.head())
    plot(data)


def plot(data):
    # data
    epochs_range = range(len(data.index))
    loss = data["loss"]
    val_loss = data["val_loss"]

    # create plot
    fig, ax = plt.subplots()

    # apply data
    ax.plot(epochs_range, loss, label="training loss", linestyle="dotted")
    ax.plot(epochs_range, val_loss, label="validation loss", linestyle="dashed")

    # draw boilerplate
    ax.set(
        xlabel="epoch", ylabel="log loss value", title="Training and Validation Loss"
    )

    caption = "Parameters: \ndropout rate = {}, initial learn rate = {},\noptimizer = {}\
, patience = {}, batch size = {}".format(
        p["dor"], p["ilr"], p["opt"], p["pat"], p["bts"]
    )
    fig.text(0.5, 0.02, caption, ha="center", style="italic")

    plt.legend(loc="upper right")

    # stylize
    ax.grid()
    fig.subplots_adjust(bottom=0.2)
    # show and export
    fig.savefig("./output.jpg")


if __name__ == "__main__":
    main()
