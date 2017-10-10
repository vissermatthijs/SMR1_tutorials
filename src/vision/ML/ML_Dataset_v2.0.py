# Load libraries
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn import datasets
import numpy as np
import csv
import pandas as pd

data=[]
target=[]

with open('plant_db.csv') as csvfile:
    dataset = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in dataset:
        data.append(row[1:])
        target.append(row[0])


data = np.asarray(data)
#print(data)

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.17, random_state=20)

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=3)
X_train_lda = lda.fit_transform(X_train_norm, Y_train)
X_test_lda = lda.transform(X_test_norm)


model = XGBClassifier()
model.fit(X_train_lda, Y_train)

print(model)

y_pred = model.predict(X_test_lda)



print ('accuracy: %0.4f' % accuracy_score(Y_test, y_pred))