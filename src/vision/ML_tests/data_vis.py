import numpy

from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix

file_name = "/home/matthijs/PycharmProjects/SMR1/src/vision/ML_tests/plant_db_v4.csv"

names = ['class', 'area', 'hull-area', 'solidity', 'perimeter', 'width', 'height', 'longest_axis', 'center-of-mass-x',
         'center-of-mass-y', 'hull_vertices', 'ellipse_center_x', 'ellipse_center_y', 'ellipse_major_axis',
         'ellipse_minor_axis', 'ellipse_angle', 'ellipse_eccentricity', 'estimated_object_count',
         'tip_points', 'tip_points_r', 'centroid_r', 'baseline_r', 'tip_number', 'vert_ave_c', 'hori_ave_c',
         'euc_ave_c',
         'ang_ave_c']
not_incl_names = ['vert_ave_b', 'hori_ave_b', 'euc_ave_b', 'ang_ave_b', 'left_lmk', 'right_lmk', 'center_h_lmk',
                  'left_lmk_r', 'right_lmk_r', 'center_h_lmk_r', 'top_lmk', 'bottom_lmk', 'center_v_lmk', 'top_lmk_r',
                  'bottom_lmk_r', 'center_v_lmk_r']

data = read_csv(file_name, names=names)
set_option('display.width', 150)
set_option('precision', 3)

description = data.describe()
print(description)

correlations = data.corr(method='pearson')
print(correlations)

skew = data.skew()
print(skew)

# Univariate Histograms
data.hist()
pyplot.show()

# Desnsity plots
data.plot(kind='density', subplots=True, layout=(6, 6), sharex=False)
pyplot.show()

# Multivariate Plots
data.plot(kind=' box ', subplots=True, layout=(6, 6), sharex=False, sharey=False)
pyplot.show()

# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0, 27, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# Scatter plot Matrix
scatter_matrix(data)
pyplot.show()
