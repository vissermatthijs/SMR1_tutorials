# Load libraries
import csv
import numpy as np
from numpy import sort

import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from xgboost import plot_importance

# ____variables____
seed = 5 #random_state
test_size = 0.26 #test_size
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

'''
lda = LinearDiscriminantAnalysis(n_components=n_components)
X_train_lda = lda.fit_transform(X_train_norm, Y_train)
X_test_lda = lda.transform(X_test_norm)

'''

# fit model no training datasad
model = XGBClassifier(
    learning_rate=0.3,
    max_depth=4,
    n_estimators=30,
    silent=False,
    objective='binary:logistic',
    nthread=-1,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.5,
    colsample_bytree=1,
    colsample_bylevel=1,
    reg_alpha=2.6,
    reg_lambda=6,
    scale_pos_weight=1,
    base_score=0.5,
    seed=0
)
#train model with data
model.fit(X_train_norm, Y_train)

# feature importance
print(model.feature_importances_)

# make predictions for test data
y_pred = model.predict(X_train_norm)

# print accuracy
print ('accuracy: %0.2f' % accuracy_score(Y_train, y_pred))

pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
plot_importance(model)
pyplot.show()

fig = plt.figure(1, figsize=(8, 6))
plt.figure(2, figsize=(8, 6))
plt.clf()

ax = Axes3D(fig, elev=-150, azim=110)
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_train_norm[:, 6], X_train_norm[:, 14], X_train_norm[:, 1], c=Y_train,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
# Plot the training points
plt.scatter(X_train_norm[:, 18], X_train_norm[:, 20], c=Y_train, cmap=plt.cm.Set1, edgecolor='k')

plt.xticks(())
plt.yticks(())

plt.show()
# split data into X and y
# split data into train and test sets
# fit model on all training data
model = XGBClassifier(
    learning_rate=0.3,
    max_depth=4,
    n_estimators=30,
    silent=False,
    objective='binary:logistic',
    nthread=-1,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.5,
    colsample_bytree=1,
    colsample_bylevel=1,
    reg_alpha=2.6,
    reg_lambda=6,
    scale_pos_weight=1,
    base_score=0.5,
    seed=0)
model.fit(X_train_norm, Y_train)
# make predictions for test data and evaluate
y_pred = model.predict(X_test)
print(y_pred)
predictions = [value for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train_norm)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, Y_train)
    # eval model
    select_X_test = selection.transform(X_test_norm)
    y_pred = selection_model.predict(select_X_test)
    predictions = [value for value in y_pred]
    accuracy = accuracy_score(Y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))
