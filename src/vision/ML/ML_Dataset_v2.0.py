# Load libraries
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import train_test_split
import numpy as np
from numpy import sort
import csv

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

# fit model no training datasad
model = XGBClassifier(
    learning_rate=0.26,
    max_depth=8,
    n_estimators=30,
    silent=False,
    objective='binary:logistic',
    nthread=-1,
    gamma=0.1,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.5,
    colsample_bytree=1,
    colsample_bylevel=1,
    reg_alpha=2.6,
    reg_lambda=5,
    scale_pos_weight=1,
    base_score=0.5,
    seed=0
)
#train model with data
model.fit(X_train_norm, Y_train)

# feature importance
print(model.feature_importances_)

# make predictions for test data
y_pred_test = model.predict(X_test_norm)
y_pred_train = model.predict(X_train_norm)


# print accuracy
print ("accuracy_on_test:"+'%0.3f' % accuracy_score(Y_test, y_pred_test))
print ("accuracy_on_train:"+'%0.3f' % accuracy_score(Y_train, y_pred_train))
