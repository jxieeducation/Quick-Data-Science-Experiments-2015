import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale


data = load_iris()
X0, y0 = data.data, data.target
y = y0[y0 != 0]
y[y == 1] = 0
y[y == 2] = 1
X = X0[y0 != 0, :]
X = scale(X)

def predict(w, x):
    wTx = np.dot(w, np.append(x, 1))
    return 1. / (1. + math.exp(- wTx ))

def update(w, x, y, alpha):
    p = predict(w, x)
    w -= alpha * (p - y) * np.append(x, 1)
    
def train(X, y, alpha, w=None):
    n, m = X.shape
    if w is None:
        w = np.zeros(m + 1)
    for j in range(10):
        for i in xrange(n):
            update(w, X[i,:], y[i], alpha)
    return w

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
Xr, yr = X[indices,], y[indices,]

w = train(Xr, yr, alpha=0.3)
p = [predict(w, x) for x in Xr]
print np.mean(yr == map(round, p))
