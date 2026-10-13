"""Load the trained models and classify PIL images. Shared by main.py, app.py and stress_test.py.

v2 (default): model2.keras, MobileNetV2 fine-tuned on gray, brain-cropped 224x224 images, 5 classes
    (the 4 tumor classes plus "other" = not a brain MRI), temperature-scaled probabilities and a
    confidence threshold below which the answer is "not sure". See train_v2.py.
v1 (legacy):  model1.h5, the original small CNN from 2023 on RGB 180x180 images, 4 classes.
"""
import json
import warnings

import numpy as np
from PIL import Image

from preprocessing import IMG_SIZE, prepare

MODEL_V2 = "model2.keras"
CALIBRATION = "calibration.json"
MODEL_V1 = "model1.h5"
CLASS_NAMES_V1 = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "Sin tumor",
    "pituitary": "Tumor pituitario",
    "other": "No es una resonancia cerebral",
}


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


class Classifier:
    """The v2 model plus its calibration. `predict(img)` returns a dict with probabilities and a verdict."""

    def __init__(self, model_path=MODEL_V2, calibration_path=CALIBRATION):
        import keras
        self.model = keras.saving.load_model(model_path, compile=False)
        cal = json.load(open(calibration_path))
        self.class_names = cal["class_names"]
        self.temperature = cal["temperature"]
        self.threshold = cal["threshold"]

    def probs(self, img):
        """Calibrated probability of each class (in self.class_names order) for a PIL image."""
        x = prepare(img)[np.newaxis].astype("float32")
        logits = self.model.predict(x, verbose=0)[0]
        return softmax(logits / self.temperature)

    def predict(self, img):
        p = self.probs(img)
        best = int(np.argmax(p))
        key = self.class_names[best]
        if key == "other":
            verdict, label = "other", "No parece una resonancia cerebral"
        elif p[best] < self.threshold:
            verdict, label = "unsure", f"No estoy seguro (lo más probable: {DISPLAY_NAMES[key]}, {p[best]:.0%})"
        else:
            verdict, label = key, DISPLAY_NAMES[key]
        return {"class": key, "confidence": float(p[best]), "verdict": verdict, "label": label,
                "probs": {c: float(v) for c, v in zip(self.class_names, p)}}


class ClassifierV1:
    """The original 2023 model, kept for comparison. Same interface as Classifier (never says 'unsure')."""

    class_names = CLASS_NAMES_V1
    threshold = 0.0

    def __init__(self, model_path=MODEL_V1):
        import keras
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*input_shape.*", category=UserWarning)
            self.model = keras.models.load_model(model_path, compile=False)

    def probs(self, img):
        x = np.asarray(img.convert("RGB").resize((180, 180), Image.Resampling.BILINEAR), dtype="float32")[np.newaxis]
        return softmax(self.model.predict(x, verbose=0)[0])

    def predict(self, img):
        p = self.probs(img)
        best = int(np.argmax(p))
        key = self.class_names[best]
        return {"class": key, "confidence": float(p[best]), "verdict": key, "label": DISPLAY_NAMES[key],
                "probs": {c: float(v) for c, v in zip(self.class_names, p)}}


def seen_by_model(img):
    """The 224x224 gray image the v2 model actually receives, as a PIL image (for the interfaces)."""
    return Image.fromarray(prepare(img)[..., 0])
