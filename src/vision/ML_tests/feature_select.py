from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

file_name = "/home/matthijs/PycharmProjects/SMR1/src/vision/ML_tests/plant_db_v4.csv"
names = ['class', 'area', 'hull-area', 'solidity', 'perimeter', 'width', 'height', 'longest_axis', 'center-of-mass-x',
         'center-of-mass-y', 'hull_vertices', 'ellipse_center_x', 'ellipse_center_y', 'ellipse_major_axis',
         'ellipse_minor_axis', 'ellipse_angle', 'ellipse_eccentricity', 'estimated_object_count',
         'tip_points', 'tip_points_r', 'centroid_r', 'baseline_r', 'tip_number', 'vert_ave_c', 'hori_ave_c',
         'euc_ave_c',
         'ang_ave_c']
data_frame = read_csv(file_name, names=names)

array = data_frame.values

X = array[:, 1:27]
Y = array[:, 0]
""" 
#feature extraction for non-negative features
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
#summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
#summarize selected features
print(features[0:5,:])
"""

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_

support = fit.support_
rank = fit.ranking_
for i in range(len(fit.support_)):
    if support[i] == True:
        print(names[i])

# PCA

# Feature Importence with Extra Trees Classifier
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
