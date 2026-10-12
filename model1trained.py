"""Load the trained model (model1.h5) and report its accuracy on the train, validation and test sets.

Usage:
    python model1trained.py

Also used as a module by main.py (load_trained_model, CLASS_NAMES, IMG_SIZE).
"""
import warnings

import numpy as np
import keras

MODEL_PATH = "model1.h5"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
# Alphabetical order of the sub-folders in Training/, which is the order Keras assigns to the labels.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def load_trained_model(path=MODEL_PATH):
    """Load the model saved by ConvolutionalTF.py.

    model1.h5 was saved with Keras 2.15. Keras 3 cannot deserialize its compile configuration,
    so the architecture and weights are loaded alone and the model is compiled again with the
    same optimizer, loss and metrics used for training.
    """
    with warnings.catch_warnings():
        # The legacy file stores an `input_shape` on its first layer; Keras 3 only warns about it.
        warnings.filterwarnings("ignore", message=".*input_shape.*", category=UserWarning)
        model = keras.models.load_model(path, compile=False)
    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model


def build_datasets():
    """Recreate the same train / validation / test splits used in ConvolutionalTF.py."""
    train_ds = keras.utils.image_dataset_from_directory(
        "Training", validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val_ds = keras.utils.image_dataset_from_directory(
        "Training", validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    test_ds = keras.utils.image_dataset_from_directory(
        "Testing", seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    model = load_trained_model()
    train_ds, val_ds, test_ds = build_datasets()
    for name, ds in [("Train", train_ds), ("Validation", val_ds), ("Test", test_ds)]:
        loss, acc = model.evaluate(ds, verbose=0)
        print(f"\n{name} accuracy: {np.round(acc, 3)}")
