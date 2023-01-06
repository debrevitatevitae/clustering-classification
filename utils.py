from typing import Tuple

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


if __name__ == '__main__':
	pass