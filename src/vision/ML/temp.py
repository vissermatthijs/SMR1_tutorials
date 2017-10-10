# Load libraries
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
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
print(data)
print(target)

#X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.17, random_state=20)

#data = np.asarray(data)
# split data into train and test sets
seed = 20
test_size = 0.25
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=test_size, random_state=seed)

print(len(X_train),len(X_test))

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=3)
X_train_lda = lda.fit_transform(X_train_norm, Y_train)
X_test_lda = lda.transform(X_test_norm)

# fit model no training data
model = XGBClassifier()
model.fit(X_train_lda, Y_train)
print(len(X_train_lda))
# make predictions for test data
y_pred = model.predict(X_test_lda)



print ('accuracy: %0.4f' % accuracy_score(Y_test, y_pred))