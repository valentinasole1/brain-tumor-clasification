import cv2 as cv
import numpy as np
import pandas as pd


def getBiggestContour(contours):
    try:
        max_cnt = contours[0]
        for cnt in contours:
            if cv.contourArea(cnt) > cv.contourArea(max_cnt):
                max_cnt = cnt
        return max_cnt
    except:
        return contours


def descriptionGenerator(filename):
    im = cv.imread(filename)
    im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    _, im = cv.threshold(im, 128, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    imContour, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    moment = cv.moments(getBiggestContour(imContour))
    huMoment = cv.HuMoments(moment)
    return huMoment


def dataset(info):
    # circulo 1, estrella 2, triangulo 3
    dataset = [
        descriptionGenerator('10triangles/t8.jpeg'),
        descriptionGenerator('10triangles/t9.jpeg'),
        descriptionGenerator('10triangles/t10.jpeg'),
        descriptionGenerator('10circles/c1.png'),
        descriptionGenerator('10circles/c2.png'),
        descriptionGenerator('10stars/e1.jpeg'),
        descriptionGenerator('10stars/e2.png'),
        descriptionGenerator('10triangles/t5.jpeg'),
        descriptionGenerator('10stars/e3.png'),
        descriptionGenerator('10circles/c3.png'),
        descriptionGenerator('10stars/e5.jpg'),
        descriptionGenerator('10stars/e6.png'),
        descriptionGenerator('10circles/c4.png'),
        descriptionGenerator('10circles/c5.png'),
        descriptionGenerator('10circles/c6.png'),
        descriptionGenerator('10stars/e7.jpg'),
        descriptionGenerator('10stars/e8.png'),
        descriptionGenerator('10triangles/t1.jpeg'),
        descriptionGenerator('10triangles/t2.jpeg'),
        descriptionGenerator('10triangles/t3.jpeg'),
        descriptionGenerator('10triangles/t4.jpeg'),
        descriptionGenerator('10stars/e9.jpg'),
        descriptionGenerator('10stars/e10.jpeg'),
        descriptionGenerator('10circles/c7.png'),
        descriptionGenerator('10circles/c8.png'),
        descriptionGenerator('10stars/e4.png'),
        descriptionGenerator('10circles/c9.jpeg'),
        descriptionGenerator('10triangles/t6.jpeg'),
        descriptionGenerator('10triangles/t7.jpeg'),
        descriptionGenerator('10circles/c10.jpeg')]
    tags = [3, 3, 3, 1, 1, 2, 2, 3, 2, 1, 2, 2, 1, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 2, 1, 3, 3, 1]

    rows = 30
    columns = 7
    for i in range(rows):
        for j in range(columns):
            dataset[i][j] = -1 * np.copysign(1.0, dataset[i][j]) * np.log10(np.absolute(dataset[i][j])) # Mapeo para agrandar la escala.
    if info == 1:
        return dataset
    if info == 0:
        return tags


def createArray (huNumber, data):
    hu = [
        data[i][huNumber-1]
        for i in range(30)
    ]
    return hu


def main():
    data = dataset(1)
    tags = dataset(0)
    col1 = "tags"
    col2 = "hu1"
    col3 = "hu2"
    col4 = "hu3"
    col5 = "hu4"
    col6 = "hu5"
    col7 = "hu6"
    col8 = "hu7"
    hu1 = createArray(1, data)
    hu2 = createArray(2, data)
    hu3 = createArray(3, data)
    hu4 = createArray(4, data)
    hu5 = createArray(5, data)
    hu6 = createArray(6, data)
    hu7 = createArray(7, data)
    data = pd.DataFrame({col1: tags, col2: hu1, col3:hu2, col4: hu3, col5: hu4, col6: hu5, col7: hu6, col8: hu7})
    data.to_excel('data.xlsx', sheet_name='sheet1', index=False)

main()

