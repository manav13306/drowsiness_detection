"""Keras version of the eye-state CNN (same architecture / weight layout as cnn_numpy.NumpyCNN)."""
import config


def build_model():
    """Output = P(eye is OPEN). Labels: closed=0, open=1. Input is float32 in [0,1]."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(config.IMG_H, config.IMG_W, 1))
    x = inputs
    for f in config.FILTERS:
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(config.DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="eye_state_cnn")
    model.compile(optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model
