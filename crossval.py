import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
import dataset as ds
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logger

log = logger.setup_logger(__name__)

def evaluate(config):
    log.info("Beginning crossvalidation...")

    kf = KFold(n_splits = 5)

    alldata, _ = ds.create_dataset(config)
    alldata = alldata.unbatch()

    foldnr = 0
    for train_ds, val_ds in kf.split(alldata):
        train_ds = train_ds.batch(batch_size = 32)
        val_ds = val_ds.batch(batch_size = 32)

        foldnr = foldnr + 1
        model = inception.create_model(config, train_ds, val_ds, kfold = foldnr)

