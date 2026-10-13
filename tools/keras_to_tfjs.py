"""Export a Keras 3 model (.keras) to the TensorFlow.js layers format (model.json + weight shard).

Usage (from the repository root):
    python tools/keras_to_tfjs.py model2.keras web/model

Also copies calibration.json (class names, temperature, confidence threshold) next to the model, because
web/index.html reads it.

Why not tensorflowjs_converter? It does not install cleanly next to TensorFlow 2.21 (protobuf conflict) and
it does not understand the Keras 3 config format anyway. TensorFlow.js reads the Keras *2* topology format,
so this script rewrites the Keras 3 config into it:
  * `inbound_nodes` with `__keras_tensor__` entries  ->  [[layer, node, tensor, {}], ...]
  * `batch_shape`                                      ->  `batch_input_shape`
  * dtype policies, initializer wrappers and the Keras 3 only keys are flattened
and writes the weights as `<layer name>/<weight name>` in one float16-quantized shard (half the size, no
measurable change in the predictions), which `tf.loadLayersModel` decodes transparently.
"""
import json
import os
import shutil
import sys

import numpy as np
import keras

# Keras 3 names every kernel "kernel"; TensorFlow.js keeps the Keras 2 names for some layers.
WEIGHT_RENAMES = {"DepthwiseConv2D": {"kernel": "depthwise_kernel"}}
DROP_KEYS = {"module", "registered_name", "build_config", "synchronized", "groups", "keepdims", "seed",
             "autocast", "quantization_config", "lora_rank", "optional"}


def clean_value(v):
    """Recursively turn Keras 3 serialized objects into the Keras 2 shape TensorFlow.js expects."""
    if isinstance(v, dict):
        if "class_name" in v and "config" in v:  # initializer / regularizer / constraint / dtype policy
            if v["class_name"] in ("DTypePolicy", "FloatDTypePolicy"):
                return v["config"].get("name", "float32")
            return {"class_name": v["class_name"], "config": clean_value(v["config"])}
        return {k: clean_value(x) for k, x in v.items() if k not in DROP_KEYS}
    if isinstance(v, list):
        return [clean_value(x) for x in v]
    return v


def convert_inbound(nodes):
    """Keras 3: [{"args": [tensor | [tensors]], "kwargs": {}}]  ->  Keras 2: [[[layer, node, idx, {}], ...]]."""
    out = []
    for node in nodes:
        refs = []
        def walk(x):
            if isinstance(x, dict) and x.get("class_name") == "__keras_tensor__":
                layer, node_index, tensor_index = x["config"]["keras_history"]
                refs.append([layer, node_index, tensor_index, {}])
            elif isinstance(x, (list, tuple)):
                for y in x:
                    walk(y)
        walk(node.get("args", []))
        out.append(refs)
    return out


def as_list_of_refs(refs):
    """Keras 3 writes a single input/output as ["layer", 0, 0]; TensorFlow.js wants [["layer", 0, 0]]."""
    return [refs] if refs and isinstance(refs[0], str) else refs


def convert_layer(layer_cfg):
    cfg = clean_value(layer_cfg["config"])
    cls = layer_cfg["class_name"]
    if cls == "InputLayer":
        shape = cfg.pop("batch_shape", None) or cfg.pop("batch_input_shape")
        cfg = {"batch_input_shape": shape, "dtype": "float32", "sparse": False, "ragged": False, "name": cfg["name"]}
    cfg.setdefault("dtype", "float32")
    if not isinstance(cfg["dtype"], str):
        cfg["dtype"] = "float32"
    return {"class_name": cls, "config": cfg, "name": cfg["name"],
            "inbound_nodes": convert_inbound(layer_cfg.get("inbound_nodes", []))}


def main(src, out):
    model = keras.saving.load_model(src, compile=False)
    config = model.get_config()
    topology = {
        "class_name": "Model",
        "config": {
            "name": config["name"],
            "layers": [convert_layer(l) for l in config["layers"]],
            "input_layers": as_list_of_refs(config["input_layers"]),
            "output_layers": as_list_of_refs(config["output_layers"]),
        },
    }
    specs, blobs = [], []
    for layer in model.layers:
        renames = WEIGHT_RENAMES.get(layer.__class__.__name__, {})
        for w in layer.weights:
            arr = np.asarray(w.numpy(), dtype=np.float32)
            specs.append({"name": f"{layer.name}/{renames.get(w.name, w.name)}", "shape": list(arr.shape),
                          "dtype": "float32", "quantization": {"dtype": "float16"}})
            blobs.append(arr.astype("<f2").tobytes())
    os.makedirs(out, exist_ok=True)
    shard = "group1-shard1of1.bin"
    with open(os.path.join(out, shard), "wb") as fh:
        for b in blobs:
            fh.write(b)
    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras v{keras.__version__}",
        "convertedBy": "tools/keras_to_tfjs.py",
        "modelTopology": {"keras_version": "2.15.0", "backend": "tensorflow", "model_config": topology},
        "weightsManifest": [{"paths": [shard], "weights": specs}],
    }
    with open(os.path.join(out, "model.json"), "w") as fh:
        json.dump(model_json, fh)
    if os.path.exists("calibration.json"):  # the page reads class names, temperature and threshold from it
        shutil.copy("calibration.json", os.path.join(out, "calibration.json"))
        print(f"copied calibration.json to {out}/")
    total = sum(int(np.prod(w["shape"])) for w in specs)
    print(f"{len(topology['config']['layers'])} layers, {len(specs)} weight tensors, {total:,} params, "
          f"shard {os.path.getsize(os.path.join(out, shard)) / 1e6:.1f} MB (float16)")
    classes = sorted({l['class_name'] for l in topology['config']['layers']})
    print("layer types:", ", ".join(classes))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
