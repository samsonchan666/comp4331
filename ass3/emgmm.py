"""
================================
EM-GMM implementation
================================
"""
# Standard scientific Python imports
import matplotlib.pyplot as plt

import scipy.io as sio
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import time

color_iter = ['gold', 'navy']

def plot_results(X, Y_, index, title):
	splot = plt.subplot(2, 1, 1 + index)
	for i, color in enumerate(color_iter, 0):  
		if not np.any(Y_ == i):
			continue
		plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
	plt.xticks(())
	plt.yticks(())
	plt.title(title)

# Calculate weight relative to cluster 1
def calweight(o, c1, c2):
	d1 = calDistSq(o, c1)
	d2 = calDistSq(o, c2)
	weight = d2/ (d2 + d1)
	return weight

# Return the euclidean distance
def calDistSq(p1, p2):
	dist = np.linalg.norm(p1-p2)
	return dist * dist

# Return the new centroid
def calCentroid(points, matrix, c):
	x = sumOfWeightSqPoint(points, matrix, c, 0) / sumOfWeightSq(matrix, c)
	y = sumOfWeightSqPoint(points, matrix, c, 1) / sumOfWeightSq(matrix, c)
	return [x,y]

def sumOfWeightSq(matrix, c):
	sum = 0
	for i in range(0, len(matrix)):
		sum += matrix[i][c] * matrix[i][c]
	return sum

def sumOfWeightSqPoint(points, matrix, c, xy):
	sum = 0
	for i in range(0, len(matrix)):
		sum += matrix[i][c] * matrix[i][c] * points[i][xy]
	return sum

'''Use sio.loadmat to load matlab data and convert into numpy.ndarray'''
mat_contents = sio.loadmat("GMM-Points")
oct_points_labeled = mat_contents['Points']

# 2d points
oct_points = np.empty([len(oct_points_labeled), 2])
oct_labels = np.empty([len(oct_points_labeled), 1])

# Split the data into points and label
for i in range(0, len(oct_points_labeled)):
	oct_points[i] = np.delete(oct_points_labeled[i], -1)
	oct_labels[i] = oct_points_labeled[i][-1]

# Start to count the time
start_time = time.time()

# Choose the first two points to be clusters
a = oct_points[0]
b = oct_points[1]
partition_matrix = np.empty([len(oct_points), 2])

# Looping of the algorithm
for loop in range(0, 10):
	# Calculate the partition matrix
	for i in range(0, len(oct_points)):
		weight = calweight(oct_points[i], a, b)
		partition_matrix[i][0] = weight
		partition_matrix[i][1] = 1 - weight
	# Update the centroid for each custer
	a = calCentroid(oct_points, partition_matrix, 0)
	b = calCentroid(oct_points, partition_matrix, 1)
	#print(partition_matrix)
	#print(a, b)

# Classify the points based the latest partition matrix
labels = np.empty([len(oct_points), 1])
for i in range(0, len(oct_points)):
	if partition_matrix[i][0] >= partition_matrix[i][1]:
		labels[i] = 1
	else:
		labels[i] = 0
#print(labels)

# Calculate the Accuracy
correct = 0
for i in range(0, len(labels)):
	if labels[i] == oct_labels[i]:
		correct += 1
accuracy = correct / len(labels)
print("Accuracy: ", accuracy)

#print the running time
print("--- %s seconds ---" % (time.time() - start_time))

# Plot the graph and GMM & original one
plot_results(oct_points, labels.flatten(), 0, 'Gaussian Mixture')
plot_results(oct_points, oct_labels.flatten(), 1, 'Original')
plt.show()

'''
test = np.asarray([[3,3], [4,10], [9,6], [14,8], [18,11], [21,7]])
a = test[0]
b = test[1]
partition_matrix = np.empty([len(test), 2])

for loop in range(0, 10):
	for i in range(0, len(test)):
		weight = calweight(test[i], a, b)
		partition_matrix[i][0] = weight
		partition_matrix[i][1] = 1 - weight
	a = calCentroid(test, partition_matrix, 0)
	b = calCentroid(test, partition_matrix, 1)
	print(partition_matrix)
	#print(a, b)

labels = np.empty([len(test), 1])
for i in range(0, len(test)):
	if partition_matrix[i][0] >= partition_matrix[i][1]:
		labels[i] = 0
	else:
		labels[i] = 1
print(labels)
'''






