import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
	np.random.seed(0)

	