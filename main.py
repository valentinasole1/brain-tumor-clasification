"""Classify one or more brain MRI images with the trained model.

Usage:
    python main.py path/to/image.jpg [more images...]
    python main.py                      # runs on the 4 sample images included in the repo

For a graphical interface run app.py instead.
"""
import argparse

import numpy as np
from PIL import Image

from model1trained import CLASS_NAMES, IMG_SIZE, load_trained_model

SAMPLE_IMAGES = ["glioma.jpg", "meningioma.jpg", "no_tumor.jpg", "pituitary.jpg"]


def preprocess(img):
    """Prepare a PIL image the same way Keras did during training: RGB, 180x180, bilinear resize.

    Grayscale MRIs are expanded to 3 identical channels. Scaling pixels to 0-1 is done by the
    Rescaling layer inside the model, so raw 0-255 values are passed here.
    """
    img = img.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype="float32")[np.newaxis, ...]  # shape (1, 180, 180, 3)


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def predict_probs(model, img):
    """Return the probability of each class (in CLASS_NAMES order) for a PIL image."""
    logits = model.predict(preprocess(img), verbose=0)[0]
    return softmax(logits)  # the model outputs logits; softmax turns them into probabilities


def img_prediction(model, image_path):
    probs = predict_probs(model, Image.open(image_path))
    idx = int(np.argmax(probs))

    print(f"\nImagen: {image_path}")
    print(f"Predicción: {CLASS_NAMES[idx]}")
    print(f"Confianza: {probs[idx]:.1%}")
    print("Probabilidades:", ", ".join(f"{c}={p:.1%}" for c, p in zip(CLASS_NAMES, probs)))
    return CLASS_NAMES[idx], float(probs[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify brain MRI images with the trained CNN.")
    parser.add_argument("images", nargs="*", help="image files to classify (default: the 4 sample images)")
    args = parser.parse_args()

    model = load_trained_model()
    for path in args.images or SAMPLE_IMAGES:
        img_prediction(model, path)
