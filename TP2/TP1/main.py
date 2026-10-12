import cv2
from TP1.funcionesTP1 import get_contours
from TP1.funcionesTP1 import threshold, denoise
from TP1.funcionesTP1 import get_trackbar_value, create_trackbar
from TP1.funciones import get_trained_model
import numpy as np
import joblib

loaded_classifier = joblib.load("../classifier.joblib")
def main():

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Window')
    create_trackbar('Threshold', 'Window', 255)
    model = get_trained_model()

    while True:
        tecla = cv2.waitKey(30)
        ret, frame = cap.read()
        frame_bn = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # frame en escala de grices
        trackbar_val = get_trackbar_value('Threshold', 'Window')
        adapt_frame = threshold(frame_bn, 255, cv2.THRESH_BINARY, trackbar_val) # devuelve la umbralización de la imagen

        frame_denoised = denoise(adapt_frame, cv2.MORPH_ELLIPSE, 5) # sacamos ruido a la imagen, MORPH_ELILIPSE tipo de elemento estructurante, y 5 el tamaño 5x5 del elemento estrcuturante

        contours = get_contours(frame_denoised, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # devuelve lista de contornos del frame_denoised. cv2.CHAIN_APPROX_NONE es el método que utiliza (todos los puntos del contorno se almacenan)  y cv2.RETR_TREE es el modo en que se encuentran los contornos almacenarlos en la estructura de un árbol jerárquico

        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                moments = cv2.moments(cnt)
                hu = cv2.HuMoments(moments)
                huMoments = [hu[cnt][0] for cnt in range(7)]
                for i in range(7):
                    huMoments[i] = -1 * np.copysign(1.0, huMoments[i]) * np.log10(np.absolute(huMoments[i]))
                    result = loaded_classifier.predict([huMoments])
                    result = int(result)
                    x, y, w, h = cv2.boundingRect(i)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if result == 1:
                        cv2.putText(frame, "Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    if result == 2:
                        cv2.putText(frame,"Star", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    if result == 3:
                        cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("cap", frame)
        if tecla == "q":
            break




