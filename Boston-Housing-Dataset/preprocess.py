import numpy as np
from sklearn.datasets import load_boston

 
boston = load_boston()
X = boston["data"]
Y = boston["target"]
size = len(boston["data"])
trainsize = 400
idx = range(size)
#shuffle the data
np.random.shuffle(idx)

def getData():
	return X[idx[:trainsize]], Y[idx[:trainsize]]

def getTestData():
	return X[idx[trainsize:]], Y[idx[trainsize:]]

