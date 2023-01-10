import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import cluster_dataset


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
	# plt.show()

	# Assign the data to two clusters
	y0 = np.zeros(n//2)
	y1 = np.ones(n//2)

	# Concatenate data and assignments
	X = np.concatenate((X0, X1), axis=0)
	y = np.concatenate((y0, y1), axis=0)
	print(y)

	# Randomly permute data and assignments
	idxs = np.random.permutation(n)
	X = X[idxs]
	y = y[idxs]

	# Split in train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	n_train = len(y_train)
	n_test = len(y_test)

	#%% Assign two points as means and see how many datapoints get correctly clustered
	means = [np.array([0., 1.]), np.array([0., -1.])]
	y_train_pred = cluster_dataset(X_train, means)
	X0 = np.array([X_train[i] for i in range(n_train) if y_train[i] == 0])
	X1 = np.array([X_train[i] for i in range(n_train) if y_train[i] == 1])
	X_c0 = np.array([X_train[i] for i in range(n_train) if y_train_pred[i] == 0])
	X_c1 = np.array([X_train[i] for i in range(n_train) if y_train_pred[i] == 1])

	fig, axs = plt.subplots(1, 2)
	axs[0].scatter(X0[:, 0], X0[:, 1], marker='o', label='cluster 0')
	axs[0].scatter(X1[:, 0], X1[:, 1], marker='o', label='cluster 1')
	axs[0].legend()
	axs[1].scatter(X_c0[:, 0], X_c0[:, 1], marker='o', label='cluster 0')
	axs[1].scatter(X_c1[:, 0], X_c1[:, 1], marker='o', label='cluster 1')
	axs[1].scatter(means[0][0], means[0][1], color='k', marker='*')
	axs[1].scatter(means[1][0], means[1][1], color='k', marker='*')
	axs[1].legend()
	plt.show()