"""Stress test: how do the models behave on images that are NOT from the Kaggle dataset?

Two groups of images, both with a known expected class:
  stress/web/        real images from Wikimedia Commons (other sequences, planes, annotations, non-MRIs).
                     stress/web/SOURCES.csv has the expected class, a note, the license and the source page.
  stress/synthetic/  dataset test images degraded the way web images usually are (overlay + arrow + caption,
                     blur, low resolution, heavy JPEG, white frame, contrast) plus a few non-MRI images.
                     Generated deterministically by this script when the folder is missing.

For every image it prints what each model says. A v2 answer counts as correct when the verdict equals the
expected class, as "unsure" when the model declines to decide, and as wrong otherwise. v1 never declines
and has no "other" class, so it is wrong by construction on non-MRI images.

Usage (from the repository root):
    python stress_test.py                 # both models, writes stress/RESULTS.md
    python stress_test.py --only v2
"""
import argparse
import csv
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SYNTH_DIR = os.path.join("stress", "synthetic")
WEB_DIR = os.path.join("stress", "web")
RESULTS = os.path.join("stress", "RESULTS.md")
SOURCES = {  # dataset test images used as the base of the synthetic variants
    "glioma": ["Testing/glioma/Te-gl_0010.jpg", "Testing/glioma/Te-gl_0114.jpg"],
    "meningioma": ["Testing/meningioma/Te-me_0129.jpg", "Testing/meningioma/Te-me_0202.jpg"],
    "notumor": ["Testing/notumor/Te-no_0013.jpg", "Testing/notumor/Te-no_0303.jpg"],
    "pituitary": ["Testing/pituitary/Te-pi_0051.jpg", "Testing/pituitary/Te-pi_0156.jpg"],
}


def annotate(img):
    """Orange semi-transparent overlay, white pointer line and a caption, like a news-site figure."""
    img = img.convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([w * 0.55, h * 0.30, w * 0.80, h * 0.55], fill=(255, 120, 0, 110))
    d.line([w * 0.80, h * 0.42, w * 0.98, h * 0.20], fill=(255, 255, 255, 255), width=max(2, w // 128))
    d.text((w * 0.03, h * 0.92), "TUMOR CEREBRAL", fill=(255, 255, 255, 255), font=ImageFont.load_default(size=max(12, h // 20)))
    return img


def frame(img):
    img = img.convert("RGB")
    w, h = img.size
    out = Image.new("RGB", (w + 2 * (w // 8), h + 2 * (h // 8)), (255, 255, 255))
    out.paste(img, (w // 8, h // 8))
    return out


VARIANTS = {
    "original": lambda im: im,
    "annotated": annotate,
    "blur": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "lowres100": lambda im: im.resize((100, int(100 * im.size[1] / im.size[0]))),
    "jpeg20": lambda im: im,  # saved with quality 20 below
    "whiteframe": frame,
    "contrast1.8": lambda im: ImageEnhance.Contrast(im.convert("RGB")).enhance(1.8),
}


def non_mri_images(rng):
    rs = np.random.RandomState(0)
    yield "noise", Image.fromarray(rs.randint(0, 256, (256, 256), dtype=np.uint8))
    yield "flat_gray", Image.new("L", (256, 256), 128)
    circle = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(circle).ellipse([48, 48, 208, 208], outline=255, width=6)
    yield "circle_drawing", circle
    grad = Image.fromarray(np.tile(np.linspace(0, 255, 256).astype(np.uint8), (256, 1)))
    yield "gradient", grad
    text = Image.new("L", (320, 200), 255)
    ImageDraw.Draw(text).text((20, 60), "Hola mundo", fill=0, font=ImageFont.load_default(size=48))
    yield "text_on_white", text


def build_synthetic():
    os.makedirs(SYNTH_DIR, exist_ok=True)
    rows = []
    for cls, files in SOURCES.items():
        for src in files:
            base = Image.open(src)
            stem = os.path.splitext(os.path.basename(src))[0]
            for vname, fn in VARIANTS.items():
                name = f"{stem}_{vname}.jpg"
                fn(base).convert("RGB").save(os.path.join(SYNTH_DIR, name), quality=20 if vname == "jpeg20" else 92)
                rows.append({"file": name, "expected": cls, "note": f"{vname} of {src}"})
    for vname, im in non_mri_images(random.Random(0)):
        name = f"nonmri_{vname}.png"
        im.save(os.path.join(SYNTH_DIR, name))
        rows.append({"file": name, "expected": "other", "note": "not an MRI"})
    with open(os.path.join(SYNTH_DIR, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "expected", "note"])
        w.writeheader(); w.writerows(rows)
    print(f"built {len(rows)} synthetic images in {SYNTH_DIR}")


def load_manifest(folder, name):
    with open(os.path.join(folder, name), newline="") as f:
        return list(csv.DictReader(f))


def describe(r):
    if r["verdict"] == "unsure":
        return f"unsure ({r['class']} {r['confidence']:.0%})"
    return f"{r['verdict']} {r['confidence']:.0%}"


def score(r, expected):
    if r["verdict"] == "unsure":
        return "unsure"
    return "ok" if r["verdict"] == expected else "wrong"


def run(models, folder, manifest, title, out):
    rows = load_manifest(folder, manifest)
    totals = {m: {"ok": 0, "unsure": 0, "wrong": 0} for m in models}
    out.append(f"\n## {title}\n")
    out.append("| image | expected | note | " + " | ".join(models) + " |")
    out.append("|---|---|---|" + "---|" * len(models))
    for row in rows:
        img = Image.open(os.path.join(folder, row["file"]))
        cells = []
        for m, clf in models.items():
            r = clf.predict(img)
            s = score(r, row["expected"])
            totals[m][s] += 1
            mark = {"ok": "✅", "unsure": "🤷", "wrong": "❌"}[s]
            cells.append(f"{mark} {describe(r)}")
        out.append(f"| {row['file']} | {row['expected']} | {row['note'][:60]} | " + " | ".join(cells) + " |")
        print(f"{row['file']:45s} {row['expected']:11s} " + "   ".join(f"{m}: {c}" for m, c in zip(models, cells)))
    n = len(rows)
    out.append("")
    for m, t in totals.items():
        out.append(f"**{m}**: {t['ok']}/{n} correct, {t['unsure']} unsure, {t['wrong']} wrong.  ")
        print(f"{title} -> {m}: {t['ok']}/{n} correct, {t['unsure']} unsure, {t['wrong']} wrong")
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the stress test on the web and synthetic image sets.")
    parser.add_argument("--only", choices=["v1", "v2"], help="run a single model")
    parser.add_argument("--v2-model", default=None, help="alternative model2.keras path")
    args = parser.parse_args()
    from classifier import Classifier, ClassifierV1

    if not os.path.exists(os.path.join(SYNTH_DIR, "MANIFEST.csv")):
        build_synthetic()
    models = {}
    if args.only != "v2":
        models["v1"] = ClassifierV1()
    if args.only != "v1":
        models["v2"] = Classifier(args.v2_model) if args.v2_model else Classifier()

    out = ["# Stress test results", "",
           "Generated by `python stress_test.py`. ✅ correct, 🤷 the model declined (below the confidence threshold), ❌ wrong.",
           "v1 = original 2023 CNN (model1.h5), v2 = MobileNetV2 with brain crop, 'other' class and calibration (model2.keras)."]
    run(models, WEB_DIR, "SOURCES.csv", "Real images from the web (Wikimedia Commons)", out)
    run(models, SYNTH_DIR, "MANIFEST.csv", "Dataset images degraded like web images, plus non-MRI images", out)
    with open(RESULTS, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nwrote {RESULTS}")
