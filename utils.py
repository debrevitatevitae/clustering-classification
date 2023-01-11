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
		np.ndarray: Shape = (n,). Cluster ID for each datapoint.
	"""
	n, _ = data.shape
	clusters = [assign_datapoint_to_cluster(data[i], means) for i in range(n)]
	return np.array(clusters)

def update_means(data:np.ndarray, clusters:np.ndarray) -> List[np.ndarray]:
	"""Divides the data in the corresponding clusters and computes the means for each cluster (update step in the k-means clustering algorithm by S. LLoyd)

	Args:
		data (np.ndarray): Shape = (n, m). Datapoints.
		clusters (np.ndarray): Cluster ID for each datapoint.

	Returns:
		List[np.ndarray]: list of means per cluster (coordinates, shape = (m,)).
	"""
	K = int(clusters.max + 1)
	means=[]
	for i in range(K):
		Xi = data[clusters==i]
		mi = np.mean(Xi, axis=0)
		means.append(mi)
	return means

def compute_scd(data:np.ndarray, clusters:np.ndarray, means:List[np.ndarray]) -> float:
	"""Compute the sum of the in-cluster distances (performance metric for k-means)

	Args:
		data (np.ndarray): Shape = (n, m). Datapoints.
		clusters (np.ndarray): Shape = (n,). Cluster ID for each datapoint.
		means (List[np.ndarray]): list of means per cluster (coordinates, shape = (m,)).

	Returns:
		float: scd value
	"""
	n, _ = data.shape
	K = int(clusters.max + 1)
	scd=0.
	for i in range(K):
		Xi = data[clusters==i]
		mi = np.mean(Xi, axis=0)
		for j in range(n):
			scd += np.linalg.norm(Xi[j]-mi)
	return scd

def lloyd_kmeans_clustering(data:np.ndarray, means_init:List[np.ndarray], tol:float=1e-4, max_iter:int=1000, verbose:bool=False) -> Tuple[List[np.ndarray], List[float]]:
	"""Performs k-means clustering following S. LLoyd's algorithm (1957) and returns the final means and the history of the sum of in-cluster distances.

	Args:
		data (np.ndarray): Shape = (n, m). Datapoints.
		means_init (List[np.ndarray]): initial list of means per cluster (coordinates, shape = (m,)).
		tol (float, optional): tolerance on the means change. Defaults to 1e-4.
		max_iter (int, optional): maximum number of Lloyd's algorithm iterations. Defaults to 1000.
		verbose (bool, optional): flag to print the scd value at every iteration. Defaults to False.

	Returns:
		Tuple[List[np.ndarray], List[float]]: final list of means per cluster, list of scd history values
	"""
	scd_history = []
	means_old = means_init
	for N in range(max_iter):

		clusters = cluster_dataset(data, means_old)
		means_new = update_means(data, clusters)

		scd_history.append(compute_scd(data, clusters, means_new))
		if verbose:
			print(f"SCD at iteration {N+1:d} = {scd_history[-1]:.4f}")
		if np.all(np.abs(np.array(means_old)-np.array(means_new)) < tol):
			print("The means change went below tolerance. Exiting the update loop...")
			break
		
		means_old = means_new
	return means_new, scd_history

if __name__ == '__main__':
	pass