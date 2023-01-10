import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generate_ellipse(n:int, axes:np.ndarray, center=np.array([0., 0.]), rot:float=0.) -> np.ndarray:
	thetas = np.random.uniform(low=0., high=2*np.pi, size=n)
	aa = np.random.uniform(low=0., high=1., size=n)
	bb = np.random.uniform(low=0, high=1., size=n)
	X = np.empty((n, 2))
	X[:, 0] = aa * axes[0] * np.cos(thetas)
	X[:, 1] = bb * axes[1] * np.sin(thetas)
	R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
	X = X @ R.T
	return X + center


if __name__ == '__main__':
	np.random.seed(0)

	#%% Generate data and split into train and test sets
	# Generate the two ellipses
	n=200
	X0 = generate_ellipse(n//2, [2., 1.])
	X1 = generate_ellipse(n//2, [2., 1.], center=[1., -2.], rot=np.pi/4)

	# Plot the ellipses
	fig, ax = plt.subplots()
	ax.scatter(X0[:, 0], X0[:, 1], marker='o', label='cluster 0')
	ax.scatter(X1[:, 0], X1[:, 1], marker='o', label='cluster 1')
	ax.legend()
	plt.show()

	# Assign the data to two clusters
	y0 = np.zeros(X0.size)
	y1 = np.ones(X1.size)

	# Concatenate data and assignments
	X = np.concatenate(X0, X1, axis=0)
	y = np.concatenate(y0, y1, axis=0)

	# Randomly permute data and assignments
	idxs = np.random.permutation(n)
	X = X[idxs]
	y = y[idxs]

	# Split in train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	#%% Cluster in k means