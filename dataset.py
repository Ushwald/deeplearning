import tensorflow as tf

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

    if not config.crossvalidation:
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
    else:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "PetImages",
            label_mode = "categorical",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )
        val_ds = "This is not a real dataset"

    if config.augmentation:
        #Do whatever is necessary for augmentation
        log.warning("Augmentation should be performed (but is not yet implemented)")

    return (train_ds, val_ds)