"""
================================
Recognizing hand-written digits
================================

An example showing how the scikit-learn can be used to recognize images of
hand-written digits.

This example is commented in the
:ref:`tutorial section of the user manual <introduction>`.

"""
print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from scipy import interpolate
from scipy import integrate
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import time
import random

'''Use sio.loadmat to load matlab data and convert into numpy.ndarray'''
#test images
mat_contents = sio.loadmat("test_images")
oct_test_images = mat_contents['test_images']
#test label
mat_contents = sio.loadmat("test_labels")
oct_test_labels = mat_contents['test_labels'][0]

#training images
mat_contents = sio.loadmat("train_images")
oct_train_images = mat_contents['train_images']
#training label
mat_contents = sio.loadmat("train_labels")
oct_train_labels = mat_contents['train_labels'][0]


#Merge test and train sets
merged_images = np.concatenate((oct_test_images, oct_train_images))
merged_labels = np.concatenate((oct_test_labels, oct_train_labels))

new_test_images = np.array([])
new_test_labels = np.array([])

new_train_images = merged_images
new_train_labels = merged_labels

y = 11001

#split the merged sets
for i in range(0,1000):
	x = random.randint(0,y)
	new_test_images = np.append(new_test_images, new_train_images[x], axis=0)
	new_train_images = np.delete(new_train_images, x, 0)
	
	new_test_labels = np.append(new_test_labels, new_train_labels[x])
	new_train_labels = np.delete(new_train_labels, x, 0)
	y = y - 1

new_test_images = np.split(new_test_images, 1000)
# print(len(new_test_images))
# print(len(new_train_images))

# print(len(new_test_labels))
# print(len(new_train_labels))

oct_train_images = new_train_images
oct_train_labels = new_train_labels

# Create a classifier: a support vector classifier
# classifier = DecisionTreeClassifier(criterion="gini", max_depth=5)
# classifier = DecisionTreeClassifier(criterion="entropy", max_depth=10)
classifier = KNeighborsClassifier(n_neighbors=5)
# classifier = svm.SVC(gamma=0.001, kernel = "poly")
# classifier = RandomForestClassifier(max_depth=10, random_state=0)
# classifier = MLPClassifier(hidden_layer_sizes=(50, ))
# classifier = MLPClassifier(hidden_layer_sizes=(100, ))
# classifier = MLPClassifier(hidden_layer_sizes=(100, 10))
# classifier = MLPClassifier(hidden_layer_sizes=(50, 20))
# classifier = MLPClassifier(alpha=1)
# classifier = GaussianNB()

#start to count the training time
start_time = time.time()
# We learn the digits from the training images
classifier.fit(oct_train_images, oct_train_labels)

# Now predict the value of the digit of the test images
# expected = oct_test_labels
# predicted = classifier.predict(oct_test_images)
expected = new_test_labels
predicted = classifier.predict(new_test_images)

#print the training time
print("--- %s seconds ---" % (time.time() - start_time))

#generate report for the classifier 
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
#calculate accuracy
print("Accuracy: %s" % accuracy_score(expected, predicted))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))




