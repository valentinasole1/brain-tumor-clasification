"""Train the shape classifier (circle=1, star=2, triangle=3) from the Hu moments stored in data.xlsx.

Usage (from the TP2/ folder):
    python trainer.py

Outputs classifier.joblib, used by predictor.py and predictorNuevo.py.
"""
import pandas as pd
from joblib import dump
from sklearn import tree
from sklearn.model_selection import train_test_split  # Import train_test_split function

HU_COLUMNS = ["hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"]


def to_float(value):
    # DescriptorGenerator.py stores each Hu moment as a 1-element array, which Excel keeps as "[0.52...]".
    return float(str(value).strip("[]"))


data = pd.read_excel("data.xlsx", sheet_name="sheet1")
y = data["tags"].astype(int).tolist()
x = [[to_float(v) for v in row] for row in data[HU_COLUMNS].itertuples(index=False)]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)  # 70% training and 30% test
classifier = tree.DecisionTreeClassifier().fit(x_train, y_train)
print("Test accuracy:", round(classifier.score(x_test, y_test), 3))
dump(classifier, 'classifier.joblib')
