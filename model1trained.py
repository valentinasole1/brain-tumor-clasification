import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers, models, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense
from matplotlib import pyplot as plt
import numpy as np

# Recrea exactamente el mismo modelo solo desde el archivo
model_mlp = keras.models.load_model('model1.h5')

dataset_path = "Training"
batch_size = 32
img_height = 180
img_width = 180
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
test_ds = tf.keras.utils.image_dataset_from_directory(
  "Testing",
  seed=123,
  image_size=(180, 180),
  batch_size=32)

# Verifique que la precisión sea la correcta
mlp_train_loss, mlp_train_acc = model_mlp.evaluate(train_ds, verbose=0)
print('\nTrain accuracy:', np.round(mlp_train_acc, 3))
mlp_val_loss, mlp_val_acc = model_mlp.evaluate(val_ds, verbose=0)
print('\nValidation accuracy:', np.round(mlp_val_acc, 3))
mlp_test_loss, mlp_test_acc = model_mlp.evaluate(test_ds, verbose=0)
print('\nTest accuracy:', np.round(mlp_test_acc, 3))

