import cv2 as cv
from joblib import load
import numpy as np
from TP1 import funcionesTP1

classifier = load('classifier.joblib')


def getContours(binary, img):
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for i in contours:
        area = cv.contourArea(i)
        if area > 0.5:
            cv.drawContours(img, i, -1, (255, 0, 0), 3)
    return contours


def main():
    cap = cv.VideoCapture(0)
    cv.namedWindow('Window')
    funcionesTP1.create_trackbar('Thresh', 'Window', 255)

    while True:
        ret, frame = cap.read()
        trackbar_val = funcionesTP1.get_trackbar_value('Thresh', 'Window')
        frame_bn = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)  # frame en escala de grices
        adapt_frame = funcionesTP1.threshold(frame_bn, 255, cv.THRESH_BINARY,
                                             trackbar_val)  # devuelve la umbralización de la imagen
        frame_denoised = funcionesTP1.denoise(adapt_frame, cv.MORPH_ELLIPSE,
                                              5)  # sacamos ruido a la imagen, MORPH_ELILIPSE tipo de elemento estructurante, y 5 el tamaño 5x5 del elemento estrcuturante

        contours = getContours(frame_denoised, frame)

        for cnt in contours:
            if cv.contourArea(cnt) > 1000:
                moments = cv.moments(cnt)
                hu = cv.HuMoments(moments)
                huMoments = [hu[i][0] for i in range(7)]
                for j in range(7):
                    huMoments[j] = -1 * np.copysign(1.0, huMoments[j]) * np.log10(np.absolute(huMoments[j]))
                result = classifier.predict([huMoments])
                result = int(result[0])
                x, y, w, h = cv.boundingRect(cnt)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if result == 1:
                    cv.putText(frame, "Circle", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                if result == 2:
                    cv.putText(frame, "Star", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                if result == 3:
                    cv.putText(frame, "Triangle", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv.imshow('Window', frame)
        cv.imshow('Denoise', frame_denoised)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


main()

