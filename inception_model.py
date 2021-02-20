import tensorflow as tf
from tensorflow import keras
import logger
from inception_custom import InceptionCustom
import pandas as pd

log = logger.setup_logger(__name__)

image_size = (180, 180)
batch_size = 32


def create_model(config, train_ds, val_ds, kfold = 0):
   

    if config.activation == 'other':
        model = InceptionCustom(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=image_size + (3,),
            pooling=None,
            classes=2,
            classifier_activation='softmax',
            act=config.activation
        )
    else:
        model = tf.keras.applications.InceptionV3(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=image_size + (3,),
            pooling=None,
            classes=2,
            classifier_activation="softmax"
        )

    #make_model(input_shape=image_size + (3,), num_classes=2)
    #keras.utils.plot_model(model, show_shapes=True)

    crossvalfoldnrstr = ''
    if config.crossvalidation:
        crossvalfoldnrstr = f"Foldnr{kfold}_"
    callbacks = [
        keras.callbacks.ModelCheckpoint("model_checkpoints/{crossvalfoldnrstr}{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}Optimizer-{config.augmentation}Augmentation_save_at_{epoch}.h5"),
    ]

    if config.optimizer == 'sgdm':
        model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=config.learningrate, momentum=config.momentum, nesterov=False, name='SGD'
                ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    else:
         model.compile(
            tf.keras.optimizers.RMSprop(
            learning_rate=config.learningrate, rho=0.9, momentum=config.momentum, epsilon=1.0, centered=False,
                name='RMSprop'
                ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    # train the model and record the history of relevant data over the epochs
    hist = model.fit(
        train_ds, epochs=config.epochs, callbacks=callbacks, validation_data=val_ds,
    )
    log.info("Training done")                                                   
                                                                                
    log.info("Saving history of loss...")                                       
    hist_df = pd.DataFrame.from_dict(hist.history)                           
    hist_df.to_csv(f"./model_training_history/{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}Optimizer-{config.augmentation}Augmentation-history.csv")  

    
    