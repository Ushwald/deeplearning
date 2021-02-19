import tensorflow as tf
from tensorflow import keras
import logger

log = logger.setup_logger(__name__)

image_size = (180, 180)
batch_size = 32


def create_model(config, train_ds, val_ds):
    #specify optimizer:
    if config.activation == 'leaky':
        #Ryan; do whatever it takes to implement leaky ReLu
        log.warning("Leaky ReLu is specified, but this is not yet implemented")

    model = tf.keras.applications.InceptionV3(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=image_size + (3,),
        pooling=None,
        classes=2,
        classifier_activation="softmax",
    )


    #make_model(input_shape=image_size + (3,), num_classes=2)
    #keras.utils.plot_model(model, show_shapes=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint("model_checkpoints/save_at_{epoch}.h5"),
    ]

    if config.optimizer == 'sgdm':
        model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=0.045, momentum=0.3, nesterov=False, name='SGD', **kwargs
                ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    else:
         model.compile(
            tf.keras.optimizers.RMSprop(
            learning_rate=0.045, rho=0.9, momentum=0.0, epsilon=1.0, centered=False,
                name='RMSprop', **kwargs
                ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    model.fit(
        train_ds, epochs=config.epochs, callbacks=callbacks, validation_data=val_ds,
    )

    
    