import cv2
import csv
import glob
import numpy
from sklearn import tree
from joblib import dump, load
import math


file_name = "/TP2"


def hu_moments_of_file(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)

    # Invert the image so the area of the UAV is filled with 1's. This is necessary since
    # cv::findContours describes the boundary of areas consisting of 1's.
    bin = 255 - bin # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

    kernel = numpy.ones((3, 3), numpy.uint8)  # Tamaño del bloque a recorrer
    # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)  # encuetra los contornos
    shape_contour = max(contours, key=cv2.contourArea)  # Agarra el contorno de area maxima

    # Calculate Moments
    moments = cv2.moments(shape_contour)  # momentos de inercia
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)  # momentos de Hu
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i])) # Mapeo para agrandar la escala.
    return huMoments


def write_hu_moments(label, writer):
    files = glob.glob('./shapes/' + label + '/*')  # label recibe el nombre de la carpeta
    hu_moments = []
    for file in files:
        hu_moments.append(hu_moments_of_file(file))
    for mom in hu_moments:
        flattened = mom.ravel()  # paso de un array de arrays a un array simple.
        row = numpy.append(flattened, label)  # le metes el flattened array y le agregas el label
        writer.writerow(row)  # Escribe una linea en el archivo.


def generate_hu_moments_file():
    with open('generated-files/shapes-hu-moments.csv', 'w',
              newline='') as file:  # Se genera un archivo nuevo (W=Write)
        writer = csv.writer(file)
        # Ahora escribo los momentos de Hu de cada uno de las figuras. Con el string "rectangle...etc" busca en la carpeta donde estan cada una de las imagenes
        # generar los momentos de Hu y los escribe sobre este archivo. (LOS DE ENTRENAMIENTO).
        write_hu_moments("5-point-star", writer)
        write_hu_moments("rectangle", writer)
        write_hu_moments("triangle", writer)

# Encargada de generar los momentos de Hu para las imagenes
#dividir


def divide(fileName):  #transforma los arrays en forma de np array
    labels = []
    moments = []
    with open(fileName) as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append(row.pop())
            moments.append(row)
    return moments, labels


def read_labels(label):
    if label == 1:
        print("5 point star")
    if label == 2:
        print("Rectangle")
    if label == 3:
        print("Triangle")


def get_trained_model():
    return load('filename.joblib')


def train_model():
    x, y = divide()
    clasificador = tree.DecisionTreeClassifier().fit(x, y)
    dump(clasificador, 'filename.joblib')


def label_to_int(string_label):
    if string_label == '5-point-star':
        return 1
    if string_label == 'rectangle':
        return 2
    if string_label == 'triangle':
        return 3

    else:
        raise Exception('unkown class_label')
