from sklearn import *
from joblib import dump
import xlwings as xw
from sklearn.model_selection import train_test_split # Import train_test_split function

ws = xw.Book("data.xlsx").sheets['sheet1']
y = ws.range("A2:A31").value
hu1 = ws.range("B2:B31").value
hu2 = ws.range("C2:C31").value
hu3 = ws.range("D2:D31").value
hu4 = ws.range("E2:E31").value
hu5 = ws.range("F2:F31").value
hu6 = ws.range("G2:G31").value
hu7 = ws.range("H2:H31").value
x = [
    [float(hu1[i].strip("[]")),
     float(hu2[i].strip("[]")),
     float(hu3[i].strip("[]")),
     float(hu4[i].strip("[]")),
     float(hu5[i].strip("[]")),
     float(hu6[i].strip("[]")),
     float(hu7[i].strip("[]"))]
    for i in range(30)
]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test
int_array = list(map(int, y_train))
classifier = tree.DecisionTreeClassifier().fit(x_train, int_array)
dump(classifier, 'classifier.joblib')

