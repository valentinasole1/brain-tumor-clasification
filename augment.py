"""Data augmentation for train_v2.py, written with plain TensorFlow ops so it runs inside tf.data.

The goal is to make the model indifferent to everything that is not the brain itself, which is exactly
what broke the v1 model on images from the web: resolution, blur, JPEG artifacts, frames, brightness and
contrast, colored overlays (gray blobs after the grayscale conversion), pointer lines, captions and noise.

augment(image) takes a float32 (H, W, C) image with values 0-255 and returns a float32 (H, W, 3) image.
"""
import math

import tensorflow as tf

P_BLUR, P_DOWNSCALE, P_JPEG, P_FRAME = 0.35, 0.35, 0.35, 0.25
P_LINES, P_BLOBS, P_PATCHES, P_NOISE = 0.35, 0.35, 0.35, 0.3


def _chance(p):
    return tf.random.uniform([]) < p


def _affine(img):
    """Random rotation (+-15 deg), zoom (0.85-1.2) and shift (+-8%), filled with black like an MRI background."""
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    angle = tf.random.uniform([], -15.0, 15.0) * math.pi / 180.0
    scale = 1.0 / tf.random.uniform([], 0.85, 1.2)
    tx = tf.random.uniform([], -0.08, 0.08) * w
    ty = tf.random.uniform([], -0.08, 0.08) * h
    cx, cy = w / 2.0, h / 2.0
    a0, a1 = scale * tf.cos(angle), -scale * tf.sin(angle)
    b0, b1 = scale * tf.sin(angle), scale * tf.cos(angle)
    a2 = cx - a0 * cx - a1 * cy + tx
    b2 = cy - b0 * cx - b1 * cy + ty
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis], transforms=transform, output_shape=tf.shape(img)[:2],
        interpolation="BILINEAR", fill_mode="CONSTANT", fill_value=0.0)
    return out[0]


def _frame(img):
    """Pad with a random constant (black, white or gray) on random sides, then resize back: web thumbnails."""
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    pads = tf.cast(tf.random.uniform([4], 0.0, 0.18) * tf.cast(tf.stack([h, h, w, w]), tf.float32), tf.int32)
    pads = pads * tf.cast(tf.random.uniform([4]) < 0.7, tf.int32)
    value = tf.random.shuffle(tf.constant([0.0, 255.0, 128.0, 200.0]))[0] + tf.random.uniform([], -20.0, 20.0)
    value = tf.clip_by_value(value, 0.0, 255.0)
    padded = tf.pad(img, [[pads[0], pads[1]], [pads[2], pads[3]], [0, 0]], constant_values=value)
    return tf.image.resize(padded, [h, w], method="bilinear")


def _downscale(img):
    """Shrink to 48-200 px and back up: low resolution thumbnails and screenshots."""
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    side = tf.random.uniform([], 48, 200, dtype=tf.int32)
    small = tf.image.resize(img, [side, side], method="bilinear", antialias=True)
    return tf.image.resize(small, [h, w], method="bilinear")


def _blur(img):
    """Gaussian blur with a random sigma (0.5-2.5 px), separable 9x9 kernel."""
    sigma = tf.random.uniform([], 0.5, 2.5)
    x = tf.range(-4.0, 5.0)
    k = tf.exp(-0.5 * (x / sigma) ** 2)
    k = k / tf.reduce_sum(k)
    c = tf.shape(img)[-1]
    kx = tf.tile(k[tf.newaxis, :, tf.newaxis, tf.newaxis], [1, 1, c, 1])
    ky = tf.tile(k[:, tf.newaxis, tf.newaxis, tf.newaxis], [1, 1, c, 1])
    out = tf.nn.depthwise_conv2d(img[tf.newaxis], kx, [1, 1, 1, 1], "SAME")
    out = tf.nn.depthwise_conv2d(out, ky, [1, 1, 1, 1], "SAME")
    return out[0]


def _photometric(img):
    """Random gamma (0.6-1.6), contrast (0.6-1.5) and brightness (+-15%)."""
    x = img / 255.0
    x = tf.pow(tf.clip_by_value(x, 0.0, 1.0), tf.random.uniform([], 0.6, 1.6))
    mean = tf.reduce_mean(x)
    x = (x - mean) * tf.random.uniform([], 0.6, 1.5) + mean
    x = x + tf.random.uniform([], -0.15, 0.15)
    return tf.clip_by_value(x, 0.0, 1.0) * 255.0


def _grid(img):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    ys, xs = tf.meshgrid(tf.range(h, dtype=tf.float32), tf.range(w, dtype=tf.float32), indexing="ij")
    return ys, xs


def _lines(img):
    """1-3 random straight segments, 1-5 px thick, mostly bright: pointer lines and arrows."""
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    ys, xs = _grid(img)
    n = tf.random.uniform([], 1, 4, dtype=tf.int32)
    for i in range(3):
        if i < n:
            p0 = tf.random.uniform([2]) * tf.stack([h, w])
            p1 = tf.random.uniform([2]) * tf.stack([h, w])
            d = p1 - p0
            length2 = tf.maximum(tf.reduce_sum(d * d), 1.0)
            t = tf.clip_by_value(((ys - p0[0]) * d[0] + (xs - p0[1]) * d[1]) / length2, 0.0, 1.0)
            dist = tf.sqrt((ys - (p0[0] + t * d[0])) ** 2 + (xs - (p0[1] + t * d[1])) ** 2)
            thickness = tf.random.uniform([], 0.5, 2.5)
            mask = tf.cast(dist <= thickness, tf.float32)[..., tf.newaxis]
            value = tf.where(_chance(0.8), tf.random.uniform([], 180.0, 255.0), tf.random.uniform([], 0.0, 120.0))
            img = img * (1.0 - mask) + value * mask
    return img


def _blobs(img):
    """1-2 semi-transparent ellipses: colored tumor overlays and circles turn into gray blobs."""
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    ys, xs = _grid(img)
    n = tf.random.uniform([], 1, 3, dtype=tf.int32)
    for i in range(2):
        if i < n:
            cy = tf.random.uniform([], 0.2, 0.8) * h
            cx = tf.random.uniform([], 0.2, 0.8) * w
            ry = tf.random.uniform([], 0.04, 0.22) * h
            rx = tf.random.uniform([], 0.04, 0.22) * w
            e = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2
            filled = _chance(0.6)
            inside = tf.cast(e <= 1.0, tf.float32)
            ring = tf.cast(tf.abs(tf.sqrt(e) - 1.0) * tf.minimum(rx, ry) <= tf.random.uniform([], 1.0, 3.0), tf.float32)
            mask = tf.where(filled, inside, ring)[..., tf.newaxis]
            alpha = tf.where(filled, tf.random.uniform([], 0.3, 0.8), 1.0)
            value = tf.random.uniform([], 0.0, 255.0)
            img = img * (1.0 - alpha * mask) + value * alpha * mask
    return img


def _patches(img):
    """1-3 small rectangles filled with noise or a flat value: captions, labels, logos."""
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    ys, xs = _grid(img)
    n = tf.random.uniform([], 1, 4, dtype=tf.int32)
    for i in range(3):
        if i < n:
            ph = tf.random.uniform([], 0.03, 0.12) * h
            pw = tf.random.uniform([], 0.08, 0.45) * w
            y0 = tf.random.uniform([]) * (h - ph)
            x0 = tf.random.uniform([]) * (w - pw)
            mask = tf.cast((ys >= y0) & (ys < y0 + ph) & (xs >= x0) & (xs < x0 + pw), tf.float32)[..., tf.newaxis]
            base = tf.random.uniform([], 0.0, 255.0)
            texture = tf.random.uniform(tf.shape(img), 0.0, 255.0)
            fill = tf.where(_chance(0.6), tf.clip_by_value(0.5 * base + 0.5 * texture, 0.0, 255.0), tf.fill(tf.shape(img), base))
            img = img * (1.0 - mask) + fill * mask
    return img


def _noise(img):
    sigma = tf.random.uniform([], 2.0, 14.0)
    return tf.clip_by_value(img + tf.random.normal(tf.shape(img), stddev=sigma), 0.0, 255.0)


def _jpeg(img):
    x = tf.image.random_jpeg_quality(tf.clip_by_value(img, 0.0, 255.0) / 255.0, 12, 90)
    return x * 255.0


def augment(img):
    """Random augmentation of one float32 image (H, W, C) in 0-255. Returns (H, W, 3) in 0-255."""
    img = tf.cast(img[..., :1], tf.float32)                # the image is gray: work on one channel
    img = tf.image.random_flip_left_right(img)
    img = _affine(img)
    img = tf.cond(_chance(P_FRAME), lambda: _frame(img), lambda: img)
    img = tf.cond(_chance(P_DOWNSCALE), lambda: _downscale(img), lambda: img)
    img = tf.cond(_chance(P_BLUR), lambda: _blur(img), lambda: img)
    img = _photometric(img)
    img = tf.cond(_chance(P_BLOBS), lambda: _blobs(img), lambda: img)
    img = tf.cond(_chance(P_LINES), lambda: _lines(img), lambda: img)
    img = tf.cond(_chance(P_PATCHES), lambda: _patches(img), lambda: img)
    img = tf.cond(_chance(P_NOISE), lambda: _noise(img), lambda: img)
    img = tf.cond(_chance(P_JPEG), lambda: _jpeg(img), lambda: img)
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = tf.ensure_shape(img, [None, None, 1])
    return tf.tile(img, [1, 1, 3])


if __name__ == "__main__":  # quick visual check: python augment.py -> data/augment_preview.png
    import sys
    import numpy as np
    from PIL import Image
    from preprocessing import prepare
    srcs = sys.argv[1:] or ["glioma.jpg", "meningioma.jpg", "no_tumor.jpg", "pituitary.jpg"]
    tiles = []
    for s in srcs:
        x = tf.constant(prepare(Image.open(s)), tf.float32)
        row = [x] + [augment(x) for _ in range(7)]
        tiles.append(np.concatenate([t.numpy().astype(np.uint8) for t in row], axis=1))
    Image.fromarray(np.concatenate(tiles, axis=0)).save("data/augment_preview.png")
    print("wrote data/augment_preview.png")
