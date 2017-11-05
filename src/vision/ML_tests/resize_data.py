from numpy import set_printoptions

from pandas import read_csv
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

file_name = '/home/matthijs/PycharmProjects/SMR1/src/vision/ML_tests/plant_db_v4.csv'

names = ['class', 'area', 'hull-area', 'solidity', 'perimeter', 'width', 'height', 'longest_axis', 'center-of-mass-x',
         'center-of-mass-y', 'hull_vertices', 'ellipse_center_x', 'ellipse_center_y', 'ellipse_major_axis',
         'ellipse_minor_axis', 'ellipse_angle', 'ellipse_eccentricity', 'estimated_object_count',
         'tip_points', 'tip_points_r', 'centroid_r', 'baseline_r', 'tip_number', 'vert_ave_c', 'hori_ave_c',
         'euc_ave_c',
         'ang_ave_c']
data_frame = read_csv(file_name, names=names)

array = data_frame.values

# Separate data into input and output components
# When? If data is comprised of attributes with varying scales
X = array[:, 1:27]
Y = array[:, 0]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
set_printoptions(precision=3)
# print every feature 5 times
print(rescaledX[0:5, :])

# Standardize data (0 mean, 1 stdev)
# When? If input data is Gaussian distributed
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
print(rescaledX[0:5, :])

# Normalize data (length of 1)
# When? If data contains lot of zeros with attributes of varying for Neuralnets and K-Nearest
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
print(normalizedX[0:5, :])

# Binarize data
# When? If you want new features or if you want 'crisp' proabliets values
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
# summarize transformed data
print(binaryX[0:5, :])
