"""Classify one or more brain MRI images from the command line.

Usage:
    python main.py path/to/image.jpg [more images...]
    python main.py                      # runs on the 5 sample images included in the repo
    python main.py --v1 image.jpg       # use the original 2023 model instead

For a graphical interface run app.py instead.
"""
import argparse

from classifier import Classifier, ClassifierV1

SAMPLE_IMAGES = ["glioma.jpg", "meningioma.jpg", "no_tumor.jpg", "pituitary.jpg", "web/examples/web_meningioma.jpg"]


def img_prediction(clf, image_path):
    from PIL import Image
    r = clf.predict(Image.open(image_path))
    print(f"\nImagen: {image_path}")
    print(f"Predicción: {r['label']}")
    print(f"Confianza: {r['confidence']:.1%}")
    print("Probabilidades:", ", ".join(f"{c}={p:.1%}" for c, p in r["probs"].items()))
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify brain MRI images with the trained model.")
    parser.add_argument("images", nargs="*", help="image files to classify (default: the sample images)")
    parser.add_argument("--v1", action="store_true", help="use the original model1.h5 instead of model2.keras")
    args = parser.parse_args()

    clf = ClassifierV1() if args.v1 else Classifier()
    for path in args.images or SAMPLE_IMAGES:
        img_prediction(clf, path)
