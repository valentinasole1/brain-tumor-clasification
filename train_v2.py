"""Train the v2 classifier: MobileNetV2 pre-trained on ImageNet, fine-tuned on the processed dataset.

Differences from the original ConvolutionalTF.py / model1.h5:
  * input: grayscale images cropped to the brain (see preprocessing.py, built by prepare_data.py)
  * a 5th class "other" (not a brain MRI) so the model can reject photos, drawings, noise...
  * heavy augmentation (augment.py): blur, JPEG, downscaling, frames, overlays, lines, captions, noise
  * transfer learning in two phases: train the new head with the base frozen, then fine-tune the top
    blocks with a low learning rate, early stopping on the validation loss
  * temperature scaling on the validation set so the probabilities are honest, and a confidence
    threshold below which the prediction is reported as "not sure"

Usage (from the repository root, after `python prepare_data.py`; about 40 min on an Apple M1 CPU):
    python train_v2.py
    python train_v2.py --epochs-head 1 --epochs-finetune 1 --limit 10     # smoke test

Outputs: model2.keras, Model2History.csv, training_curves_v2.png, calibration.json
"""
import argparse
import csv
import json
import os

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from augment import augment
from preprocessing import IMG_SIZE

DATA_DIR = os.path.join("data", "processed")
MODEL_PATH = "model2.keras"
HISTORY_PATH = "Model2History.csv"
CURVES_PATH = "training_curves_v2.png"
CALIBRATION_PATH = "calibration.json"
BATCH_SIZE = 32
SEED = 123
FINE_TUNE_FROM = "block_10_expand"   # layers from here to the top are trained in phase 2
AUTOTUNE = tf.data.AUTOTUNE


def load_split(subset, limit=None):
    ds = keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"), validation_split=0.2, subset=subset, seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=None, color_mode="rgb")
    class_names = ds.class_names
    if subset == "training":
        ds = ds.shuffle(1024, seed=SEED).map(lambda x, y: (augment(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if limit:
        ds = ds.take(limit)
    ds = ds.prefetch(AUTOTUNE)
    ds.class_names = class_names
    return ds


def load_test(limit=None):
    ds = keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"), shuffle=False, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, color_mode="rgb")
    class_names = ds.class_names
    if limit:
        ds = ds.take(limit)
    ds = ds.prefetch(AUTOTUNE)
    ds.class_names = class_names
    return ds


def build_model(n_classes):
    """One flat functional model (no nested sub-model), which keeps the TensorFlow.js export simple."""
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(inputs)   # raw 0-255 -> -1..1
    base = keras.applications.MobileNetV2(input_tensor=x, include_top=False, weights="imagenet")
    x = layers.GlobalAveragePooling2D(name="pool")(base.output)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(n_classes, name="logits")(x)
    model = keras.Model(inputs, outputs, name="brain_tumor_v2")

    # Sanity check: the ImageNet weights really went into our flat model.
    reference = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for name in ("Conv1", "block_16_project", "Conv_1"):
        np.testing.assert_array_equal(model.get_layer(name).get_weights()[0], reference.get_layer(name).get_weights()[0])
    return model


def set_trainable(model, phase):
    """phase 'head': only the classifier trains. phase 'finetune': also the top blocks, BatchNorm stays frozen."""
    train_from = [l.name for l in model.layers].index(FINE_TUNE_FROM)
    for i, layer in enumerate(model.layers):
        if phase == "head":
            layer.trainable = layer.name == "logits"
        else:
            layer.trainable = i >= train_from and not isinstance(layer, layers.BatchNormalization)
    model.compile(optimizer=keras.optimizers.Adam(1e-3 if phase == "head" else 1e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])


def logits_and_labels(model, ds):
    logits, labels = [], []
    for x, y in ds:
        logits.append(model.predict_on_batch(x))
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def fit_temperature(logits, labels):
    """Temperature scaling: the single scalar T that minimizes the NLL of softmax(logits / T) on validation."""
    best_t, best_nll = 1.0, np.inf
    for t in np.arange(0.5, 5.01, 0.05):
        p = softmax(logits / t)
        nll = -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return round(best_t, 2)


def report(name, logits, labels, class_names, temperature, threshold):
    probs = softmax(logits / temperature)
    pred = probs.argmax(1)
    conf = probs.max(1)
    acc = (pred == labels).mean()
    print(f"\n{name}: accuracy {acc:.3%} on {len(labels)} images (T={temperature})")
    print("  per-class recall:", ", ".join(f"{c}={(pred[labels == i] == i).mean():.1%}" for i, c in enumerate(class_names)))
    cm = np.zeros((len(class_names), len(class_names)), int)
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    print("  confusion matrix (rows = true, cols = predicted):")
    for c, row in zip(class_names, cm):
        print(f"    {c:>11} " + " ".join(f"{v:5d}" for v in row))
    print("  threshold  coverage  accuracy-when-confident")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        keep = conf >= thr
        acc_conf = (pred[keep] == labels[keep]).mean() if keep.any() else float("nan")
        print(f"    {thr:.2f}     {keep.mean():6.1%}     {acc_conf:6.1%}")
    keep = conf >= threshold
    return {"accuracy": round(float(acc), 4), "coverage_at_threshold": round(float(keep.mean()), 4),
            "accuracy_when_confident": round(float((pred[keep] == labels[keep]).mean()), 4) if keep.any() else None}


def plot_history(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = range(1, len(rows) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(epochs, [r["accuracy"] for r in rows], label="train")
    a1.plot(epochs, [r["val_accuracy"] for r in rows], label="validation")
    a1.set_title("Accuracy"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(epochs, [r["loss"] for r in rows], label="train")
    a2.plot(epochs, [r["val_loss"] for r in rows], label="validation")
    a2.set_title("Loss"); a2.set_xlabel("epoch"); a2.legend()
    fig.tight_layout(); fig.savefig(CURVES_PATH, dpi=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the v2 brain tumor classifier (MobileNetV2).")
    parser.add_argument("--epochs-head", type=int, default=4)
    parser.add_argument("--epochs-finetune", type=int, default=14)
    parser.add_argument("--threshold", type=float, default=0.7, help="confidence below which the app says 'not sure'")
    parser.add_argument("--limit", type=int, default=None, help="use only this many batches per split (smoke test)")
    parser.add_argument("--out", default=MODEL_PATH)
    args = parser.parse_args()
    keras.utils.set_random_seed(SEED)

    train_ds, val_ds, test_ds = load_split("training", args.limit), load_split("validation", args.limit), load_test(args.limit)
    class_names = train_ds.class_names
    print("classes:", class_names)
    model = build_model(len(class_names))
    print(f"params: {model.count_params():,}")

    history = []
    def record(h, phase):
        for i in range(len(h.history["loss"])):
            history.append({"phase": phase, **{k: float(v[i]) for k, v in h.history.items()}})

    print(f"\nPhase 1: training the classifier head ({args.epochs_head} epochs)")
    set_trainable(model, "head")
    record(model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head), "head")

    print(f"\nPhase 2: fine-tuning from {FINE_TUNE_FROM} (up to {args.epochs_finetune} epochs, early stopping)")
    set_trainable(model, "finetune")
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1)]
    record(model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune, callbacks=callbacks), "finetune")

    with open(HISTORY_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "loss", "accuracy", "val_loss", "val_accuracy", "learning_rate"], extrasaction="ignore")
        w.writeheader(); w.writerows(history)
    plot_history(history)
    export = keras.Model.from_config(model.get_config())   # same architecture and weights, no optimizer state
    export.set_weights(model.get_weights())
    export.save(args.out)
    print(f"\nsaved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB), {HISTORY_PATH}, {CURVES_PATH}")

    val_logits, val_labels = logits_and_labels(model, val_ds)
    temperature = fit_temperature(val_logits, val_labels)
    val_stats = report("Validation", val_logits, val_labels, class_names, temperature, args.threshold)
    test_logits, test_labels = logits_and_labels(model, test_ds)
    test_stats = report("Test", test_logits, test_labels, class_names, temperature, args.threshold)
    calibration = {"class_names": class_names, "temperature": temperature, "threshold": args.threshold,
                   "image_size": IMG_SIZE, "validation": val_stats, "test": test_stats}
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\nwrote {CALIBRATION_PATH}: {json.dumps(calibration)}")
