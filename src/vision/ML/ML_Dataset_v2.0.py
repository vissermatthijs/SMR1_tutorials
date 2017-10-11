# Load libraries
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import csv

# ____variables____
seed =26 #random_state
test_size = 0.24 #test_size
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


lda = LinearDiscriminantAnalysis(n_components=n_components)
X_train_lda = lda.fit_transform(X_train_norm, Y_train)
X_test_lda = lda.transform(X_test_norm)


# fit model no training data
model = XGBClassifier()
model.fit(X_train_lda, Y_train)

# feature importance
print(model.feature_importances_)

# make predictions for test data
y_pred = model.predict(X_test_lda)

plt.scatter(X_train[:, 6], X_train_lda[:, 2], c=Y_train, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

print ('accuracy: %0.4f' % accuracy_score(Y_test, y_pred))