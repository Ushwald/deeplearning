import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

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
    else:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                "PetImages",
                validation_split=val_split,
                subset="training",
                label_mode = "categorical",
                seed=1337,
                image_size=image_size,
                batch_size=batch_size,
            )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "PetImages",
            validation_split=val_split,
            subset="validation",
            label_mode = "categorical",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )

    if config.augmentation:
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(0.1),
            ]
        )

        augmented_train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y))

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    return (train_ds, val_ds)