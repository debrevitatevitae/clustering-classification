import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from utils import compute_pca, plot_singular_values


if __name__ == '__main__':
	np.random.seed(0)

	#%% Load dataset and split
	iris = load_iris()
	X = iris.data
	y = iris.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

	#%% Plot the first 3 features of the training data, distinguishing the labels
	# 0: setosa, 1: versicolour, 2: virginica
	setosa = y_train==0
	versicolour = y_train==1
	virginica = y_train==2
	
	X_setosa = X_train[setosa]
	X_versicolour = X_train[versicolour]
	X_virginica = X_train[virginica]
	y_setosa = y_train[setosa]
	y_versicolour = y_train[versicolour]
	y_virginica = y_train[virginica]

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(X_setosa[:,0], X_setosa[:,1], X_setosa[:,2], y_setosa, label='setosa')
	ax.scatter(X_versicolour[:,0], X_versicolour[:,1], X_versicolour[:,2], y_versicolour, label='versicolour')
	ax.scatter(X_virginica[:,0], X_virginica[:,1], X_virginica[:,2], y_virginica, label='virginica')
	ax.set_xlabel('Sepal length')
	ax.set_ylabel('Sepal width')
	ax.set_zlabel('Petal length')
	ax.set_title('Training data')
	ax.legend()
	plt.show()

	#%% Do the PCA of the train data and plot the singular values
	U, s, VT = compute_pca(X_train)
	V = VT.T
	plot_singular_values(s)
	print(U.shape)
	print(s.shape)
	print(VT.shape)

	#%% Plot the data along the first two principal components
	X_pc2 = V[:, :2]

	X_pc2_setosa = X_pc2[setosa]
	X_pc2_versicolour = X_pc2[versicolour]
	X_pc2_virginica = X_pc2[virginica]
	print(len(X_pc2_setosa))
	print(len(X_pc2_versicolour))
	print(len(X_pc2_virginica))

	fig, ax = plt.subplots()
	ax.scatter(X_pc2_setosa[:,0], X_pc2_setosa[:,1], label='setosa')
	ax.scatter(X_pc2_versicolour[:,0], X_pc2_versicolour[:,1], label='versicolour')
	ax.scatter(X_pc2_virginica[:,0], X_pc2_virginica[:,1], label='virginica')
	ax.set_xlabel('PC 1')
	ax.set_ylabel('PC 2')
	ax.set_title('Training data along the first two principal components')
	ax.legend()
	plt.show()
