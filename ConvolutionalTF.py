"""Train a CNN that classifies brain MRIs into 4 classes: glioma, meningioma, notumor, pituitary.

Usage:
    python ConvolutionalTF.py

Expects the dataset in Training/<class>/*.jpg and Testing/<class>/*.jpg (see README.md).
Outputs:
    model1.h5            trained model
    Model1History.csv    accuracy / loss per epoch
    training_curves.png  accuracy and loss plots
"""
import numpy as np
import keras
from keras import layers, Sequential
from keras.callbacks import CSVLogger
from matplotlib import pyplot as plt

# ------- Building the datasets -------
dataset_path = "Training"
batch_size = 32
img_height = 180
img_width = 180
train_ds = keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
val_ds = keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
test_ds = keras.utils.image_dataset_from_directory(
    "Testing",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names  # alphabetical order of the sub-folders: glioma, meningioma, notumor, pituitary
num_classes = len(class_names)
print("Classes:", class_names)

# ---- Model building -----
# Pixel normalization (0-255 -> 0-1) is done by the Rescaling layer inside the model,
# so exactly the same preprocessing is applied when training and when predicting.
model_CNN = Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)  # The last layer has to have as many neurons as the quantity of classes
])
# ----- Compile the model ----
model_CNN.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
model_CNN.summary()

epochs = 12
csv_logger = CSVLogger('Model1History.csv')  # Store the model training in a .csv

# Train the model
history = model_CNN.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[csv_logger]
)

# Save the model
model_CNN.save('model1.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# ------- Plot the model progression --------
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training_curves.png')
plt.show()

# ----- Testing the model -----
train_loss, train_acc = model_CNN.evaluate(train_ds, verbose=0)
print('\nTrain accuracy:', np.round(train_acc, 3))
val_loss, val_acc = model_CNN.evaluate(val_ds, verbose=0)
print('\nValidation accuracy:', np.round(val_acc, 3))
test_loss, test_acc = model_CNN.evaluate(test_ds, verbose=0)
print('\nTest accuracy:', np.round(test_acc, 3))
