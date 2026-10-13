"""Build the processed dataset used by train_v2.py.

Every image of Training/ and Testing/ goes through preprocessing.prepare_gray (grayscale, crop to the brain,
pad to a square, 224x224) and is written as a PNG to data/processed/{train,test}/<class>/. The same pipeline
runs at prediction time, so the model never sees the original framing, resolution or color.

It also creates a 5th class, "other", with images that are not brain MRIs, so the model can say
"this is not an MRI" instead of forcing one of the four tumor classes:
  * photos from Imagenette (a 10-class subset of ImageNet, 160 px, from fast.ai:
    https://github.com/fastai/imagenette), read straight from the downloaded .tgz
  * synthetic images: noise, flat colors, gradients, geometric shapes, text, stripes

Usage (from the repository root, takes a couple of minutes):
    python prepare_data.py                       # downloads Imagenette to data/raw/ if needed
    python prepare_data.py --photos path/to/imagenette2-160.tgz
"""
import argparse
import io
import os
import random
import ssl
import string
import tarfile
import urllib.request
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from preprocessing import prepare_gray

RAW_DIRS = {"train": "Training", "test": "Testing"}
OUT_DIR = os.path.join("data", "processed")
PHOTOS_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"   # 94 MB, 13k photos
N_OTHER = {"train": (1000, 400), "test": (250, 100)}  # (photos, synthetic) per split
SEED = 123


def convert_one(args):
    src, dst = args
    try:
        prepare_gray(Image.open(src)).save(dst, optimize=True)
        return None
    except Exception as e:  # a corrupt file should not kill the whole run
        return f"{src}: {e}"


def convert_mri_folders(workers):
    jobs = []
    for split, raw in RAW_DIRS.items():
        for cls in sorted(os.listdir(raw)):
            if cls.startswith("."):
                continue
            out = os.path.join(OUT_DIR, split, cls)
            os.makedirs(out, exist_ok=True)
            for name in sorted(os.listdir(os.path.join(raw, cls))):
                if name.startswith("."):
                    continue
                dst = os.path.join(out, os.path.splitext(name)[0] + ".png")
                if not os.path.exists(dst):
                    jobs.append((os.path.join(raw, cls, name), dst))
    print(f"MRI images to convert: {len(jobs)}")
    with Pool(workers) as pool:
        errors = [e for e in pool.imap_unordered(convert_one, jobs, chunksize=16) if e]
    for e in errors:
        print("  skipped", e)


def download_photos(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading Imagenette (94 MB)...")
        ctx = ssl.create_default_context()
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
        with urllib.request.urlopen(PHOTOS_URL, context=ctx) as r, open(path, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    return path


def photo_images(path, split, n, rng):
    """n random Imagenette photos (train/ folder for 'train', val/ folder for 'test') as PIL RGB images."""
    prefix = f"imagenette2-160/{'train' if split == 'train' else 'val'}/"
    with tarfile.open(path, "r:gz") as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile() and m.name.startswith(prefix) and m.name.endswith(".JPEG")),
                         key=lambda m: m.name)
        chosen = rng.sample(members, n)
        return [Image.open(io.BytesIO(tar.extractfile(m).read())).convert("RGB") for m in chosen]


def synthetic_image(rng):
    """A random non-MRI image: noise, flat color, gradient, shapes, text or stripes."""
    size = rng.choice([64, 128, 224, 320, 480])
    w, h = size, rng.choice([size, int(size * 0.75), int(size * 1.33)])
    kind = rng.choice(["noise", "noise", "flat", "gradient", "shapes", "shapes", "text", "stripes"])
    bg = tuple(rng.randrange(256) for _ in range(3))
    if kind == "noise":
        arr = np.random.RandomState(rng.randrange(1 << 30)).randint(0, 256, (h, w, 3 if rng.random() < .5 else 1), dtype=np.uint8)
        img = Image.fromarray(np.repeat(arr, 3, axis=2) if arr.shape[2] == 1 else arr)
        if rng.random() < .5:
            img = img.resize((max(4, w // rng.choice([4, 8, 16])), max(4, h // rng.choice([4, 8, 16])))).resize((w, h), Image.Resampling.NEAREST)
        return img
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    if kind == "flat":
        return img
    if kind == "gradient":
        c2 = tuple(rng.randrange(256) for _ in range(3))
        horizontal = rng.random() < .5
        for i in range(w if horizontal else h):
            t = i / max(1, (w if horizontal else h) - 1)
            c = tuple(int(bg[k] * (1 - t) + c2[k] * t) for k in range(3))
            if horizontal:
                d.line([(i, 0), (i, h)], fill=c)
            else:
                d.line([(0, i), (w, i)], fill=c)
        return img
    if kind == "stripes":
        step = rng.randrange(4, max(5, w // 4))
        c2 = tuple(rng.randrange(256) for _ in range(3))
        for i in range(0, w + h, step * 2):
            d.line([(i, 0), (i - h, h)] if rng.random() < .5 else [(0, i), (w, i - w)], fill=c2, width=step)
        return img
    if kind == "text":
        for _ in range(rng.randrange(1, 6)):
            txt = "".join(rng.choice(string.ascii_letters + string.digits + " ") for _ in range(rng.randrange(3, 20)))
            font = ImageFont.load_default(size=rng.randrange(10, max(12, h // 3)))
            d.text((rng.randrange(w), rng.randrange(h)), txt, fill=tuple(rng.randrange(256) for _ in range(3)), font=font)
        return img
    for _ in range(rng.randrange(1, 7)):  # shapes
        c = tuple(rng.randrange(256) for _ in range(3))
        x0, y0 = rng.randrange(w), rng.randrange(h)
        x1, y1 = rng.randrange(w), rng.randrange(h)
        box = [min(x0, x1), min(y0, y1), max(x0, x1) + 1, max(y0, y1) + 1]
        shape = rng.choice(["ellipse", "rectangle", "line", "polygon"])
        if shape == "ellipse":
            d.ellipse(box, fill=c if rng.random() < .7 else None, outline=c, width=rng.randrange(1, 6))
        elif shape == "rectangle":
            d.rectangle(box, fill=c if rng.random() < .7 else None, outline=c, width=rng.randrange(1, 6))
        elif shape == "line":
            d.line([x0, y0, x1, y1], fill=c, width=rng.randrange(1, 8))
        else:
            d.polygon([(rng.randrange(w), rng.randrange(h)) for _ in range(rng.randrange(3, 7))], fill=c)
    return img


def make_other(photos_path):
    for split, (n_photos, n_synth) in N_OTHER.items():
        out = os.path.join(OUT_DIR, split, "other")
        os.makedirs(out, exist_ok=True)
        if len(os.listdir(out)) >= n_photos + n_synth:
            continue
        rng = random.Random(SEED if split == "train" else SEED + 1)
        images = photo_images(photos_path, split, n_photos, rng) + [synthetic_image(rng) for _ in range(n_synth)]
        for i, img in enumerate(images):
            prepare_gray(img).save(os.path.join(out, f"other_{i:04d}.png"), optimize=True)
        print(f"{split}/other: {len(images)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--photos", default=os.path.join("data", "raw", "imagenette2-160.tgz"),
                        help="path to imagenette2-160.tgz (downloaded here if missing)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    convert_mri_folders(args.workers)
    make_other(download_photos(args.photos))
    for split in RAW_DIRS:
        counts = {c: len(os.listdir(os.path.join(OUT_DIR, split, c))) for c in sorted(os.listdir(os.path.join(OUT_DIR, split)))}
        print(split, counts)
