"""Image preprocessing shared by the v2 model (training, CLI, Gradio app and stress test).

The browser version in web/index.html re-implements exactly the same steps in JavaScript, so keep
both in sync if you change anything here.

Pipeline for one image:
  1. grayscale (an MRI carries no color information, so colored overlays stop being a clue)
  2. find the brain: on a small blurred copy, take the pixels that differ from the background (the
     median of the border) by more than DELTA, morphological opening to drop thin lines and text,
     largest connected component, bounding box with a small margin
  3. crop to that box, pad to a square with black, resize to IMG_SIZE x IMG_SIZE
  4. replicate the gray channel 3 times (MobileNetV2 expects 3 channels), raw 0-255 values
"""
import numpy as np
from PIL import Image

IMG_SIZE = 224      # model input size
MASK_SIZE = 128     # longest side of the small copy used to locate the brain
MARGIN = 0.04       # extra border around the brain box, fraction of the box size
MIN_AREA = 0.05     # if the largest blob is smaller than this fraction of the image, do not crop
DELTA = 25          # a pixel belongs to the head when it differs from the background by more than this (0-255)


def to_gray(img):
    """PIL image -> uint8 array (H, W). Uses PIL's ITU-R 601-2 luma: L = 0.299 R + 0.587 G + 0.114 B."""
    return np.asarray(img.convert("L"), dtype=np.uint8)


def _box_blur3(a):
    p = np.pad(a, 1, mode="edge")
    h, w = a.shape
    return sum(p[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)) / 9.0


def _erode(m):
    p = np.pad(m, 1, constant_values=False)
    h, w = m.shape
    out = np.ones_like(m)
    for dy in range(3):
        for dx in range(3):
            out &= p[dy:dy + h, dx:dx + w]
    return out


def _dilate(m):
    p = np.pad(m, 1, constant_values=False)
    h, w = m.shape
    out = np.zeros_like(m)
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + h, dx:dx + w]
    return out


def _largest_component(mask):
    """Largest 4-connected blob of a small boolean mask. Returns (blob mask, area in pixels)."""
    h, w = mask.shape
    flat = mask.ravel()
    labels = np.zeros(h * w, dtype=np.int32)
    best_label, best_area, label = 0, 0, 0
    for start in np.flatnonzero(flat):
        if labels[start]:
            continue
        label += 1
        labels[start] = label
        stack, area = [int(start)], 0
        while stack:
            i = stack.pop()
            area += 1
            y, x = divmod(i, w)
            if y > 0 and flat[i - w] and not labels[i - w]:
                labels[i - w] = label; stack.append(i - w)
            if y < h - 1 and flat[i + w] and not labels[i + w]:
                labels[i + w] = label; stack.append(i + w)
            if x > 0 and flat[i - 1] and not labels[i - 1]:
                labels[i - 1] = label; stack.append(i - 1)
            if x < w - 1 and flat[i + 1] and not labels[i + 1]:
                labels[i + 1] = label; stack.append(i + 1)
        if area > best_area:
            best_area, best_label = area, label
    return (labels == best_label).reshape(h, w), best_area


def brain_box(gray):
    """Bounding box (x0, y0, x1, y1), x1/y1 exclusive, of the brain in a gray uint8 image, or None.

    Works on a copy whose longest side is MASK_SIZE pixels. The background level is the median of the
    border pixels; if it is light (figure on a white page) the dark pixels are the foreground instead.
    """
    H, W = gray.shape
    scale = MASK_SIZE / max(H, W)
    mw, mh = max(1, round(W * scale)), max(1, round(H * scale))
    small = Image.fromarray(gray).resize((mw, mh), Image.Resampling.BILINEAR)
    small = _box_blur3(np.asarray(small, dtype=np.float64))
    s8 = np.clip(np.rint(small), 0, 255).astype(np.uint8)
    border = np.concatenate([s8[0], s8[-1], s8[:, 0], s8[:, -1]])
    bg = float(np.median(border))                        # the background is whatever touches the border
    mask = (s8 > bg + DELTA) if bg < 128 else (s8 < bg - DELTA)
    mask = _dilate(_dilate(_erode(_erode(mask))))        # opening: removes thin lines and text
    blob, area = _largest_component(mask)
    if area < MIN_AREA * mw * mh:
        return None
    ys, xs = np.nonzero(blob)
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mx, my = MARGIN * (bx1 - bx0), MARGIN * (by1 - by0)
    x0 = max(0, int(np.floor((bx0 - mx) / scale)))
    y0 = max(0, int(np.floor((by0 - my) / scale)))
    x1 = min(W, int(np.ceil((bx1 + mx) / scale)))
    y1 = min(H, int(np.ceil((by1 + my) / scale)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def crop_and_square(gray, box):
    """Crop a gray array to `box` and pad it to a square with black, keeping the content centered."""
    if box is not None:
        x0, y0, x1, y1 = box
        gray = gray[y0:y1, x0:x1]
    h, w = gray.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = gray
    return canvas


def prepare(img, size=IMG_SIZE):
    """PIL image -> uint8 array (size, size, 3): gray, cropped to the brain, squared, 3 identical channels."""
    gray = to_gray(img)
    square = crop_and_square(gray, brain_box(gray))
    g = np.asarray(Image.fromarray(square).resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)
    return np.stack([g, g, g], axis=-1)


def prepare_gray(img, size=IMG_SIZE):
    """Same as prepare() but returns a single-channel PIL image (used to write the processed dataset)."""
    gray = to_gray(img)
    square = crop_and_square(gray, brain_box(gray))
    return Image.fromarray(square).resize((size, size), Image.Resampling.BILINEAR)
