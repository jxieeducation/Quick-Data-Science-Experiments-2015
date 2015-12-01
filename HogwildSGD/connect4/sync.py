import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.preprocessing import scale
from pandas import Series, DataFrame
import pandas as pd

data = pd.read_csv('connect-4.data')
X, y = data.drop(['win'], axis=1), data['win']
y[y == 'win'] = 1
y[y == 'draw'] = 0
y[y == 'loss'] = 1
X[X == 'b'] = 0
X[X == 'x'] = -1
X[X == 'o'] = 1
X = X.as_matrix()
y = y.as_matrix()

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

w = train(Xr, yr, alpha=0.1)
p = [predict(w, x) for x in Xr]
print np.mean(yr == map(round, p))
