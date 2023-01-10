from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def subtract_average(X:np.ndarray) -> np.ndarray:
	"""Subtracts the per-feature averages from the data

	Args:
		X (np.ndarray): Data. Shape = (n, m)

	Returns:
		np.ndarray: subtracted-average data. Shape = (m, n)
	"""
	n, _ = X.shape
	# compute the averages for each of the features
	x_avg = np.mean(X, axis=0)
	return X - np.outer(np.ones(n), x_avg)

def compute_pca(X:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute the PCA, using SVD.

	Args:
		X (np.ndarray): data. Shape = (n, m)

	Returns:
		Tuple[np.ndarray, np.ndarray, np.ndarray]: U, S, V^T
	"""
	n, _ = X.shape
	B = subtract_average(X)
	return np.linalg.svd(B.T / np.sqrt(n), full_matrices=False)

def plot_singular_values(s:np.ndarray) -> None:
	_, axs = plt.subplots(1, 2)
	axs[0].semilogy(s, 'b-')
	axs[0].set(xlabel='id', ylabel='singular value', title='Singular values')
	axs[1].plot(np.cumsum(s), 'b-')
	axs[1].set(xlabel='id', ylabel='cumsum', title="Cumulative sum of singular values")
	plt.show()

def assign_datapoint_to_cluster(point:np.ndarray, means:List[np.ndarray]) -> int:
	"""Assigns a datapoint to one among K clusters.

	Args:
		point (np.ndarray): Shape = (n,). Datapoint.
		means (List[np.ndarray]): list of K means (coordinates, shape = (m,)).

	Returns:
		int: cluster between 0 and K to which the datapoint is assigned.
	"""
	d_min = np.linalg.norm(point - means[0])
	cluster = 0
	for i, m in enumerate(means[1:]):
		d = np.linalg.norm(point - m)
		if d < d_min:
			cluster = i+1
			d_min = d
	return cluster

def cluster_dataset(data:np.ndarray, means:List[np.ndarray]) -> np.ndarray:
	"""Assign each point in a data array to one of the clusters defined by the means.

	Args:
		data (np.ndarray): Shape = (n, m). Datapoints.
		means (List[np.ndarray]): list of K means (coordinates, shape = (m,)).

	Returns:
		np.ndarray: Shape = (n,). Clusters for each datapoint.
	"""
	n, _ = data.shape
	clusters = [assign_datapoint_to_cluster(data[i], means) for i in range(n)]
	return np.array(clusters)

if __name__ == '__main__':
	pass