import cv2
from PIL import Image
import tensorflow as tf
import numpy as np

from model1trained import train_ds

glioma = '/Users/martuvillar/glioma.jpg'
no_tumor = '/Users/martuvillar/sano.jpg'
pituitary = '/Users/martuvillar/pitutiary.jpg'
meningioma = 'Users/martuvillar/meningioma.jpg'

model = tf.keras.models.load_model('model1.h5')

def img_prediction(image):

    img = Image.open(image)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (180, 180))
    img = img.reshape(1, 180, 180, 3)
    p = model.predict(img)

    class_names = train_ds.class_names

    predicted_class = np.argmax(p[0])
    confidence = p[0][predicted_class]

    predicted_label = class_names[predicted_class]
    print("Predicción:", predicted_label)
    print("Confianza:", confidence)
    print("P", p)


img_prediction(pituitary)
