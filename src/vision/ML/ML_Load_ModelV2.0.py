# Load libraries
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import joblib
import csv

# load model from file
loaded_model = joblib.load("pima.joblib.dat")

# ____variables____
seed = 5 #random_state
test_size = 0.21 #test_size
n_components = 3 #LDA components

data=[]
target=[]

with open('plant_db.csv') as csvfile:
    dataset = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in dataset:
        data.append(row[1:])
        target.append(row[0])

data = np.asarray(data)

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=test_size, random_state=seed)

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# feature importance
print(loaded_model.feature_importances_)

# make predictions for test data
y_pred_test = loaded_model.predict(X_test_norm)
y_pred_train = loaded_model.predict(X_train_norm)

print ("accuracy_on_test:"+'%0.3f' % accuracy_score(Y_test, y_pred_test))
print ("accuracy_on_train:"+'%0.3f' % accuracy_score(Y_train, y_pred_train))