"""
================================
DBSCAN implementation
================================
"""
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import performance metrics
# from sklearn import datasets, svm, metrics
#from scipy import interpolate
#from scipy import integrate
import scipy.io as sio
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import time


def findNeighors(set, P, eps):
	_neighbors = []
	for p2 in range(0, len(set)):
		# Calculate the euclidean distance
		dist = np.linalg.norm(set[P]-set[p2])
		if (dist < eps):
			_neighbors.append(p2)
	return _neighbors


'''Use sio.loadmat to load matlab data and convert into numpy.ndarray'''
mat_contents = sio.loadmat("DBSCAN-Points")
# 2d points
oct_points = mat_contents['Points']

'''
db = DBSCAN(eps=0.12, min_samples=3).fit(oct_points)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
'''

# Start to count the time
start_time = time.time()

labels = np.zeros(len(oct_points), dtype=int)
core_samples_mask = np.zeros(len(oct_points), dtype=bool)

i = 1
eps = 0.12
min_samples = 3
for p1 in range(0, len(oct_points)):
	# Skip if the point had been visited and labled
	if not (labels[p1] == 0):
		continue
	# Find the neighbors points based on the epsilon
	neighbors = findNeighors(oct_points, p1, eps)

	# Noise point
	if len(neighbors) < min_samples:
	    labels[p1] = -1
	# Not a noise point
	else: 
		labels[p1] = i
		# Core point
		core_samples_mask[p1] = True
		j = 0
		# Start to grow cluster with respect to the neighbors
		while j < len(neighbors) :
			pn = neighbors[j]
			if labels[pn] == -1:
				# Border point
				labels[pn] = i
			elif labels[pn] == 0:
				labels[pn] = i
				# Find neighbors of the neighbor point
				pn_neighbors = findNeighors(oct_points, pn, eps)
				if len(pn_neighbors) >= min_samples:
					# Core point
					core_samples_mask[pn] = True
					neighbors = neighbors + pn_neighbors
			j += 1
		i += 1

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

'''
print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"
  % metrics.silhouette_score(oct_points, labels))
'''

#print the running time
print("--- %s seconds ---" % (time.time() - start_time))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
      for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
	if k == -1:
	    # Black used for noise.
		col = [0, 0, 0, 1]

	class_member_mask = (labels == k)
	xy = oct_points[class_member_mask & core_samples_mask]
	plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
	         markeredgecolor='k', markersize=14)

	xy = oct_points[class_member_mask & ~core_samples_mask]
	plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
	         markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()







