"""Web interface: upload a brain MRI and see the predicted tumor type.

Usage:
    python app.py

Then open the URL printed in the terminal (http://127.0.0.1:7860 by default).
"""
import gradio as gr

from main import SAMPLE_IMAGES, predict_probs
from model1trained import CLASS_NAMES, load_trained_model

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "Sin tumor",
    "pituitary": "Tumor pituitario",
}

model = load_trained_model()  # loaded once, when the app starts


def classify(img):
    if img is None:
        return {}
    probs = predict_probs(model, img)
    return {DISPLAY_NAMES[c]: float(p) for c, p in zip(CLASS_NAMES, probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", sources=["upload", "clipboard"], label="Resonancia magnética (MRI)"),
    outputs=gr.Label(num_top_classes=4, label="Clasificación"),
    examples=[[path] for path in SAMPLE_IMAGES],
    title="Clasificación de tumores cerebrales",
    description=(
        "Subí una resonancia magnética cerebral y el modelo (una CNN entrenada con TensorFlow/Keras) "
        "la clasifica en glioma, meningioma, tumor pituitario o sin tumor. "
        "También podés probar con los ejemplos de abajo."
    ),
    article=(
        "Trabajo final de *Procesamiento de Imágenes y Señales Biomédicas*, Valentina Sole y Martina Villar. "
        "Proyecto académico: no reemplaza un diagnóstico médico."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
