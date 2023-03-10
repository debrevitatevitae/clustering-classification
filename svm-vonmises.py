import os
import sys
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.svm as svm

from utils import count_misclassifications


def generate_vm_data_plane_principal_stresses(sig_yield:float, sig_ult:float, n:int=100) -> Tuple[np.ndarray, np.ndarray]:
    """Samples the uniform distribution below the ultimate strength in plane principal components and returns labeled data according to the Von Mises criterion. 

    Args:
        sig_yield (float): yield strenght of the material
        sig_ult (float): ultimate strenght of the material
        n (int, optional): number of samples. Defaults to 100.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X, y
    """
    X = np.empty((n, 2))
    y = np.empty((n,))
    for i in range(n):
        sig = np.random.uniform(low=-sig_ult, high=sig_ult, size=(2,))
        y[i] = 0 if sig[0]**2 -sig[0]*sig[1] + sig[1]**2 - sig_yield**2 < 0. else 1
        X[i, :] = sig
    return X, y

def plot_vm_data_plane_principal_stresses(axis:Axes, X:np.ndarray, y:np.ndarray, title='Principal stress states classified with Von Mises criterion', labels=['intact', 'yielded']):
    X_intact = X[y==0]
    X_failed = X[y==1]
    axis.scatter(X_intact[:,0], X_intact[:,1], label=labels[0])
    axis.scatter(X_failed[:,0], X_failed[:,1], label=labels[1])
    axis.set_xlabel('sig_1 (MPa)')
    axis.set_ylabel('sig_2 (MPa)')
    axis.set_title(title)
    axis.legend()


if __name__ == '__main__':
    np.random.seed(0)

    #%% Load data, split into train and test sets and plot train set
    # Material: aluminum alloy 2024
    sig_yield = 42  # MPa
    sig_ultimate = 64  # MPa
    X, y = generate_vm_data_plane_principal_stresses(sig_yield, sig_ultimate, n=200)
    fig, ax = plt.subplots()
    plot_vm_data_plane_principal_stresses(ax, X, y)
    fig.tight_layout()
    # plt.show()

    #%% Split the data in a train and test set and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = MinMaxScaler(feature_range=(-1., 1.))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #%% Fit an svm and predict on both train and test sets
    clf  = svm.SVC(C=1., kernel='poly', degree=2, gamma='scale')
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    #%% Plot the real data and predictions for train and test sets
    fig, axs = plt.subplots(2, 2)
    plot_vm_data_plane_principal_stresses(axs[0, 0], X_train, y_train, title='Train set, true')
    plot_vm_data_plane_principal_stresses(axs[0, 1], X_train, y_train_pred, title='Train set, predicetd')
    plot_vm_data_plane_principal_stresses(axs[1, 0], X_test, y_test, title='Test set, true')
    plot_vm_data_plane_principal_stresses(axs[1, 1], X_test, y_test_pred, title='Test set, predicted')
    fig.suptitle("VM data: cfr with predictions of SVM, poly kernel, deg = 2", fontsize=12)
    fig.tight_layout()
    # plt.show()

    #%% Count the number of misclassified points
    wrong_train = count_misclassifications(y_train_pred, y_train)
    wrong_test = count_misclassifications(y_test_pred, y_test)
    print(f"Number of misclassifications in the train set: {wrong_train:d}/{len(y_train):d}")
    print(f"Number of misclassifications in the test set: {wrong_test:d}/{len(y_test):d}")