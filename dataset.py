import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np


import logger
import os

log = logger.setup_logger(__name__)

def create_dataset(config, val_split = 0.2):
    log.info("Loading dataset...")

    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("PetImages", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)
    image_size = (180, 180)
    batch_size = 32

    if config.crossvalidation:
        log.warning("Crossvalidation is used, but not implemented in this way, turn around now!")    
    kf = KFold(n_splits = 5)

    filenames = []
    labels = []
      

    for file in os.listdir('PetImages/Cat'):
        filenames.append(os.path.join('Cat', file))
        labels.append('Cat')

    for file in os.listdir('PetImages/Dog'):
        filenames.append(os.path.join('Dog', file))
        labels.append('Dog')


    d = {'filename': filenames, 'label': labels}
    alldata = pd.DataFrame(d)
    alldata = alldata.sample(frac=1).reset_index(drop=True) 
    Y = alldata[['label']]

    idg = ImageDataGenerator(width_shift_range=0.0,
                         height_shift_range=0.0,
                         zoom_range=0.0,
                         fill_mode='nearest',
                         horizontal_flip = False,
                         rescale=None)


    for train_index, val_index in kf.split(np.zeros(len(Y)), Y):
        training_data = alldata.iloc[train_index]
        validation_data = alldata.iloc[val_index]

        train_ds = idg.flow_from_dataframe(training_data, target_size = (180, 180), directory = 'PetImages', x_col = "filename", y_col = "label", class_mode = "categorical", shuffle = True)
        val_ds  = idg.flow_from_dataframe(validation_data, target_size = (180, 180), directory = 'PetImages', x_col = "filename", y_col = "label", class_mode = "categorical", shuffle =True)
        break
    #augmentation:

    if config.augmentation:
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(0.1),
            ]
        )

        augmented_git atrain_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y))
        train_ds = augmented_train_ds


    return (train_ds, val_ds)