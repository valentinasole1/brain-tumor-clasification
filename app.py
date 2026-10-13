"""Web interface: upload a brain MRI and see the predicted tumor type.

Usage:
    python app.py

Then open the URL printed in the terminal (http://127.0.0.1:7860 by default).
"""
import gradio as gr

from classifier import DISPLAY_NAMES, Classifier, seen_by_model
from main import SAMPLE_IMAGES

clf = Classifier()  # loaded once, when the app starts


def classify(img):
    if img is None:
        return "", {}, None
    r = clf.predict(img)
    if r["verdict"] == "other":
        text = f"{r['label']}. Probabilidad {r['confidence']:.0%} de que no sea una resonancia de cerebro, así que no se clasifica el tumor."
    elif r["verdict"] == "unsure":
        text = f"{r['label']}. Por debajo del umbral de confianza ({clf.threshold:.0%}): la imagen probablemente se aleja de las del entrenamiento."
    else:
        text = f"{r['label']} (confianza {r['confidence']:.1%})."
    return text, {DISPLAY_NAMES[c]: p for c, p in r["probs"].items()}, seen_by_model(img)


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", sources=["upload", "clipboard"], label="Resonancia magnética (MRI)"),
    outputs=[
        gr.Textbox(label="Resultado"),
        gr.Label(num_top_classes=5, label="Probabilidades"),
        gr.Image(label="Lo que ve el modelo (gris, recortada al cerebro, 224×224)", height=224),
    ],
    examples=[[path] for path in SAMPLE_IMAGES],
    title="Clasificación de tumores cerebrales",
    description=(
        "Subí una resonancia magnética cerebral y el modelo (MobileNetV2 ajustada con TensorFlow/Keras) "
        "la clasifica en glioma, meningioma, tumor pituitario o sin tumor. Si la imagen no parece una resonancia "
        "de cerebro, o el modelo no está seguro, lo dice. También podés probar con los ejemplos de abajo."
    ),
    article=(
        "Trabajo final de *Procesamiento de Imágenes y Señales Biomédicas*, Valentina Sole y Martina Villar. "
        "Proyecto académico: no reemplaza un diagnóstico médico."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
