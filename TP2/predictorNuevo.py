import cv2
import TP1.funcionesTP1
from joblib import load
import numpy as np

classifier = load("classifier.joblib")


def main():

    cap = cv2.VideoCapture(0)

    cv2.namedWindow('Window')
    TP1.funcionesTP1.create_trackbar('Threshold', 'Window', 255)
    TP1.funcionesTP1.create_trackbar('Min Area', 'Window', 10000)
    TP1.funcionesTP1.create_trackbar('Max Area', 'Window', 99999)

    while True:
        ret, frame = cap.read()
        frame_bn = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # frame en escala de grices
        trackbar_val = TP1.funcionesTP1.get_trackbar_value('Threshold', 'Window')

        trackbar_min_val = TP1.funcionesTP1.get_trackbar_value('Min Area', 'Window')
        trackbar_max_val = TP1.funcionesTP1.get_trackbar_value('Max Area', 'Window')

        adapt_frame = TP1.funcionesTP1.threshold(frame_bn, 255, cv2.THRESH_BINARY, trackbar_val) # devuelve la umbralización de la imagen

        frame_denoised = TP1.funcionesTP1.denoise(adapt_frame, cv2.MORPH_ELLIPSE, 5) # sacamos ruido a la imagen, MORPH_ELILIPSE tipo de elemento estructurante, y 5 el tamaño 5x5 del elemento estrcuturante

        contours = TP1.funcionesTP1.get_contours(frame_denoised, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # devuelve lista de contornos del frame_denoised. cv2.CHAIN_APPROX_NONE es el método que utiliza (todos los puntos del contorno se almacenan)  y cv2.RETR_TREE es el modo en que se encuentran los contornos almacenarlos en la estructura de un árbol jerárquico

        filtered_contours = TP1.funcionesTP1.filter_contours_byArea(contours, trackbar_min_val, trackbar_max_val) # devuelve una lista de contornos que cumplen con que su area este entre min area y max area

        for cnt in filtered_contours:
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
            if cv2.contourArea(cnt) > 1000:
                moments = cv2.moments(cnt)
                hu = cv2.HuMoments(moments)
                huMoments = [hu[i][0] for i in range(7)]
                for j in range(7):
                    huMoments[j] = -1 * np.copysign(1.0, huMoments[j]) * np.log10(np.absolute(huMoments[j]))
                result = classifier.predict([huMoments])
                result = int(result[0])
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if result == 1:
                    cv2.putText(frame, "Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                if result == 2:
                    cv2.putText(frame, "Star", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                if result == 3:
                    cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('Window', frame)
        cv2.imshow('Denoise', frame_denoised)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

main()
