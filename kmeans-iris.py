import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from utils import compute_pca, cluster_dataset, lloyd_kmeans_clustering


def plot_iris_2_pcs(axis, X, y, title='Iris dataset'):
	_, _, VT = compute_pca(X)
	V = VT.T
	X_pc2 = V[:, :2]
	X_pc2_setosa = X_pc2[y==0]
	X_pc2_versicolour = X_pc2[y==1]
	X_pc2_virginica = X_pc2[y==2]
	axis.scatter(X_pc2_setosa[:,0], X_pc2_setosa[:,1], label='setosa')
	axis.scatter(X_pc2_versicolour[:,0], X_pc2_versicolour[:,1], label='versicolour')
	axis.scatter(X_pc2_virginica[:,0], X_pc2_virginica[:,1], label='virginica')
	axis.set_xlabel('PC 1')
	axis.set_ylabel('PC 2')
	axis.set_title(title)
	axis.legend()
	return

def add_mean_2_pcs_to_plot(axis, mean, label=None):
	_, _, VT = compute_pca(mean.reshape(-1, 1))
	V = VT.T
	mean_pc2 = V[:2, 0]
	axis.scatter(mean_pc2[0], mean_pc2[1], color='k', marker='*', label=label)
	axis.legend()
	return



if __name__ == '__main__':
	np.random.seed(0)

	#%% Load data and split into train and test sets
	iris = load_iris()
	X = iris.data
	n, m = X.shape
	y = iris.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	#%% Plot training and test sets divided in labels
	fig, axs = plt.subplots(1, 2)
	plot_iris_2_pcs(axs[0], X_train, y_train, title="Train points of Iris")
	plot_iris_2_pcs(axs[1], X_test, y_test, title="Test points of Iris")
	fig.tight_layout()
	# plt.show()

	#%% Define three radom means for the initial clusters and plot them
	means_0 = [np.random.uniform(low=X_train.min(axis=0), high=X_train.max(axis=0)) for _ in range(m)]
	y_0_train = cluster_dataset(X_train, means_0)
	y_0_test = cluster_dataset(X_test, means_0)

	fig, axs = plt.subplots(1, 2)
	plot_iris_2_pcs(axs[0], X_train, y_0_train, title="Train points of Iris clustered with random mean",)
	plot_iris_2_pcs(axs[1], X_test, y_0_test, title="Test points of Iris clustered with random mean",)
	for i in range(m):
		add_mean_2_pcs_to_plot(axs[0], means_0[i], label='means')
		add_mean_2_pcs_to_plot(axs[1], means_0[i], label='means')
	fig.tight_layout()
	plt.show()
