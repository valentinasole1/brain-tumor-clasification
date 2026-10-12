"""Convert the Keras 2 .h5 model to the TensorFlow.js layers format (model.json + weight shard).

Usage (from the repository root):
    python tools/h5_to_tfjs.py model1.h5 web/model

Only needs h5py and numpy (both installed with tensorflow). The output is what web/index.html loads.
"""
import h5py, numpy as np

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
f = h5py.File(src, "r")
cfg = json.loads(f.attrs["model_config"])
mw = f["model_weights"]

def s(x): return x.decode() if isinstance(x, bytes) else x

specs, blobs = [], []
for ln in [s(n) for n in mw.attrs["layer_names"]]:
    g = mw[ln]
    for wn in [s(w) for w in g.attrs.get("weight_names", [])]:
        arr = np.asarray(g[wn], dtype="<f4")
        specs.append({"name": wn.split(":")[0], "shape": list(arr.shape), "dtype": "float32"})
        blobs.append(arr.tobytes())

shard = "group1-shard1of1.bin"
with open(os.path.join(out, shard), "wb") as fh:
    for b in blobs: fh.write(b)

model_json = {
    "format": "layers-model",
    "generatedBy": f"keras v{s(f.attrs['keras_version'])}",
    "convertedBy": "h5_to_tfjs.py (manual conversion)",
    "modelTopology": {
        "keras_version": s(f.attrs["keras_version"]),
        "backend": s(f.attrs["backend"]),
        "model_config": cfg,
    },
    "weightsManifest": [{"paths": [shard], "weights": specs}],
}
with open(os.path.join(out, "model.json"), "w") as fh:
    json.dump(model_json, fh)

total = sum(int(np.prod(w["shape"])) for w in specs)
print(f"weights: {len(specs)} tensors, {total:,} params, shard {os.path.getsize(os.path.join(out, shard))/1e6:.1f} MB")
for w in specs: print("  ", w["name"], w["shape"])
