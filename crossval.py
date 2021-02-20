import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
import inception_model as inception
import dataset as ds
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logger

log = logger.setup_logger(__name__)


def evaluate(config):
    log.info("Beginning crossvalidation...")

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
    Y = alldata[['label']]
    log.info(alldata)

    #Any augmentation could be performed here
    if config.augmentation:
        log.warning("The augmentation is desired, but not yet implemented for crossvalidation")
    idg = ImageDataGenerator(width_shift_range=0.0,
                         height_shift_range=0.0,
                         zoom_range=0.0,
                         fill_mode='nearest',
                         horizontal_flip = False,
                         rescale=None)

    foldnr = 0
    for train_index, val_index in kf.split(np.zeros(len(Y)),Y):
        training_data = alldata.iloc[train_index]
        validation_data = alldata.iloc[val_index]

        train_data_generator = idg.flow_from_dataframe(training_data, target_size = (180, 180), directory = 'PetImages', x_col = "filename", y_col = "label", class_mode = "categorical", shuffle = False)
        val_data_generator  = idg.flow_from_dataframe(validation_data, target_size = (180, 180), directory = 'PetImages', x_col = "filename", y_col = "label", class_mode = "categorical", shuffle =False)
	

        foldnr = foldnr + 1
        model = inception.create_model(config, train_data_generator, val_data_generator , kfold = foldnr)

