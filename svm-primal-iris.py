import os
import sys
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.svm as svm

from utils import compute_pca


def plot_iris_2_class_2_pcs(axis, X, y, U, title='Iris dataset: first two classes', labels=['setosa', 'versicolor']):
    X_pc2 = np.transpose(U.T @ X.T)
    # truncate to 2 components
    X_pc2 = X_pc2[:, :2]
    X_pc2_setosa = X_pc2[y==0]
    X_pc2_versicolor = X_pc2[y==1]
    axis.scatter(X_pc2_setosa[:,0], X_pc2_setosa[:,1], label=labels[0])
    axis.scatter(X_pc2_versicolor[:,0], X_pc2_versicolor[:,1], label=labels[1])
    axis.set_xlabel('PC 1')
    axis.set_ylabel('PC 2')
    axis.set_title(title)
    axis.legend()
    return

def hyperplane_2pcs(w:np.ndarray, b:float, U:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the hyperplanes coordinates into the two principal components of the training data.

    Args:
        w (np.ndarray): coordinates of the normal to the separating hyperplane. Shape = (1, m)
        b (float): intercept of the hyperplane.
        U (np.ndarray): matrix of principal components. Shape = (m, m).

    Returns:
        Tuple[np.ndarray, np.ndarray]: the set of two coordinates in principal components for the points on the hyperplane.
    """
    # Compute the principal components
    _, m = w.shape
    w_pcs = U.T @ w.reshape(m)
    # Truncate to 2 principal components
    w_pcs = w_pcs[:2]
    # Compute the pc-coordinates of the points on the hyperplane
    X_h_pc1 = np.linspace(-2., 2., 100)
    X_h_pc2 = - w_pcs[0]/w_pcs[1] * X_h_pc1 - b
    return X_h_pc1, X_h_pc2

def plot_hyperplane_2pcs(axis, w, b, U):
    X_h_pc1, X_h_pc2 = hyperplane_2pcs(w, b,U)
    axis.plot(X_h_pc1, X_h_pc2, 'k-', label='hyperplane')
    axis.set_xlabel('PC 1')
    axis.set_ylabel('PC 2')
    axis.legend()
    return

def plot_supp_vecs_2pcs(axis, X_sv, U):
    X_sv_pc = np.transpose(U.T @ X_sv.T)
    X_sv_pc2 = X_sv_pc[:, :2]
    axis.scatter(X_sv_pc2[:, 0], X_sv_pc2[:, 1], s=100, facecolor="None", edgecolor='red', label='supp. vecs')
    axis.set_xlabel('PC 1')
    axis.set_ylabel('PC 2')
    axis.legend()
    return
	

if __name__ == '__main__':
    np.random.seed(0)

    #%% Load data, split into train and test sets and plot train set
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Extract the first two classes
    idxs_c1_c2 = np.isin(y, [0, 1])
    X = X[idxs_c1_c2]
    y = y[idxs_c1_c2]
    m, n = X.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    U, s, VT = compute_pca(X_train)
    fig, axs = plt.subplots(1, 2)
    plot_iris_2_class_2_pcs(axs[0], X_train, y_train, U, title="Train points of Iris")
    plot_iris_2_class_2_pcs(axs[1], X_test, y_test, U, title="Test points of Iris")
    fig.tight_layout()
    # plt.show()

    #%% Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #%% Fit the svm
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train_scaled, y_train)
    w_opt = classifier.coef_
    b_opt = classifier.intercept_

    #%% Plot the separating hyperplane on the principal components
    fig, axs = plt.subplots(1, 2)
    plot_iris_2_class_2_pcs(axs[0], X_train_scaled, y_train, U, title="Train points of Iris with hyperplane")
    plot_hyperplane_2pcs(axs[0], w_opt, b_opt, U)
    axs[0].set_ylim([-2.5, 2.5])
    plot_iris_2_class_2_pcs(axs[1], X_test_scaled, y_test, U, title="Test points of Iris")
    plot_hyperplane_2pcs(axs[1], w_opt, b_opt, U)
    axs[1].set_ylim([-2.5, 2.5])
    fig.tight_layout()
    # plt.show()

    #%% Plot the support vectors on the principal components
    fig, ax = plt.subplots()
    plot_iris_2_class_2_pcs(ax, X_train_scaled, y_train, U, title="Train points of Iris with hyperplane and SVs")
    plot_hyperplane_2pcs(ax, w_opt, b_opt, U)
    ax.set_ylim([-2.5, 2.5])
    plot_supp_vecs_2pcs(ax, classifier.support_vectors_, U)
    plt.show()