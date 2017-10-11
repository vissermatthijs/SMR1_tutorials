import csv
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

""" 
Step one: Read in the data from the yucca's
Data will consist off 15 features:
area,                   0
hull_area,              1
solidity,               2    
perimeter,              3
width,                  4
height,                 5   
longest_axis,           6
center-of-mass-x,       7    -> probably no influence             
center-mass-y,          8    -> probably no influence
hull_vertices,          9
ellipse_center_x,       10
ellipse_center_y,       11
ellipse_major_axis,     12
ellipse_minor_axis,     13
ellipse_angle,          14
ellipse_eccentricity    15

missing hull verticies
"""

data = []
target = []

with open('plant_db.csv') as csvfile:
    dataset = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in dataset:
        data.append(row[1:])
        target.append(row[0])
data = np.asarray(data)

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0, random_state=20)
print(data)
print(len(X_train[:, 2]))
# Plot the training points
plt.scatter(X_train[:, 6], X_train[:, 15], c=Y_train, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()
