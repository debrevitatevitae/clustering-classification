import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.svm as svm

from utils import compute_pca


def plot_iris_2_class_2_pcs(axis, X, y, title='Iris dataset: first two classes', labels=['setosa', 'versicolor']):
	_, _, VT = compute_pca(X)
	V = VT.T
	X_pc2 = V[:, :2]
	X_pc2_setosa = X_pc2[y==0]
	X_pc2_versicolor = X_pc2[y==1]
	axis.scatter(X_pc2_setosa[:,0], X_pc2_setosa[:,1], label=labels[0])
	axis.scatter(X_pc2_versicolor[:,0], X_pc2_versicolor[:,1], label=labels[1])
	axis.set_xlabel('PC 1')
	axis.set_ylabel('PC 2')
	axis.set_title(title)
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

    fig, axs = plt.subplots(1, 2)
    plot_iris_2_class_2_pcs(axs[0], X_train, y_train, title="Train points of Iris")
    plot_iris_2_class_2_pcs(axs[1], X_test, y_test, title="Test points of Iris")
    fig.tight_layout()
    plt.show()

    #%% Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    #%% Fit the svm
    classifier = svm.LinearSVC()
    classifier.fit(X_train_scaled, y_train)
    w_opt = classifier.coef_
    print(w_opt)

    #%% Plot the separating hyperplane
    