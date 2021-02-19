import tensorflow as tf
from tensorflow import keras

image_size = (180, 180)
batch_size = 32

def create_model(config, train_ds, val_ds):
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
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, epochs=config.epochs, callbacks=callbacks, validation_data=val_ds,
    )
