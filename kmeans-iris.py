import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from utils import compute_pca, cluster_dataset, lloyd_kmeans_clustering


def plot_iris_2_pcs(axis, X, y, title='Iris dataset', labels=['setosa', 'versicolour', 'virginica']):
	_, _, VT = compute_pca(X)
	V = VT.T
	X_pc2 = V[:, :2]
	X_pc2_setosa = X_pc2[y==0]
	X_pc2_versicolour = X_pc2[y==1]
	X_pc2_virginica = X_pc2[y==2]
	axis.scatter(X_pc2_setosa[:,0], X_pc2_setosa[:,1], label=labels[0])
	axis.scatter(X_pc2_versicolour[:,0], X_pc2_versicolour[:,1], label=labels[1])
	axis.scatter(X_pc2_virginica[:,0], X_pc2_virginica[:,1], label=labels[2])
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
	K = 3

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	#%% Plot training and test sets divided in labels
	fig, axs = plt.subplots(1, 2)
	plot_iris_2_pcs(axs[0], X_train, y_train, title="Train points of Iris")
	plot_iris_2_pcs(axs[1], X_test, y_test, title="Test points of Iris")
	fig.tight_layout()
	# plt.show()

	#%% Define three radom means for the initial clusters and plot them
	means_0 = [np.random.uniform(low=X_train.min(axis=0), high=X_train.max(axis=0)) for _ in range(K)]
	y_0_train = cluster_dataset(X_train, means_0)
	y_0_test = cluster_dataset(X_test, means_0)

	fig, axs = plt.subplots(1, 2)
	plot_iris_2_pcs(axs[0], X_train, y_0_train, title="Train points of Iris clustered with random mean", labels=['cluster 1', 'cluster 2', 'cluster 3'])
	plot_iris_2_pcs(axs[1], X_test, y_0_test, title="Test points of Iris clustered with random mean", labels=['cluster 1', 'cluster 2', 'cluster 3'])
	# for i in range(K):
	# 	label = 'means' if i == m-1 else None
	# 	add_mean_2_pcs_to_plot(axs[0], means_0[i], label=label)
	# 	add_mean_2_pcs_to_plot(axs[1], means_0[i], label=label)
	fig.tight_layout()
	# plt.show()

	#%% Start from the three random means and cluster with Lloyd's algorithm
	means_final, scd_history = lloyd_kmeans_clustering(X_train, means_0, verbose=True)

	fig, ax = plt.subplots()
	ax.plot(scd_history)
	ax.set_xlabel('iteration')
	ax.set_ylabel('SCD')
	ax.set_title('SCD during k-means algorithm iteration. Clustering of Iris.', fontdict={'fontsize': 12})
	# plt.show()

	# Cluster according to the final means
	y_f_train = cluster_dataset(X_train, means_final)
	y_f_test = cluster_dataset(X_test, means_final)

	fig, axs = plt.subplots(1, 2)
	plot_iris_2_pcs(axs[0], X_train, y_f_train, title="Train points of Iris after Lloyd's clustering", labels=['cluster 1', 'cluster 2', 'cluster 3'])
	plot_iris_2_pcs(axs[1], X_test, y_f_test, title="Test points of Iris after Lloyd's clustering", labels=['cluster 1', 'cluster 2', 'cluster 3'])
	# for i in range(K):
	# 	label = 'means' if i == m-1 else None
	# 	add_mean_2_pcs_to_plot(axs[0], means_final[i], label=label)
	# 	add_mean_2_pcs_to_plot(axs[1], means_final[i], label=label)
	fig.tight_layout()
	plt.show()
