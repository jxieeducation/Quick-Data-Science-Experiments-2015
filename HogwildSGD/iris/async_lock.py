import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
import ctypes
import multiprocessing as mp
from multiprocessing import Process, Value, Lock

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

def update(w, x, y, alpha, lock):
    new_w = np.frombuffer(w.get_obj())
    p = predict(new_w, x)
    with lock:
        new_w -= alpha * (p - y) * np.append(x, 1)

def batch(w, X, y, alpha, lock):
    for i in xrange(X.shape[0]):
        update(w, X[i,:], y[i], alpha, lock)
    
def train(X, y, alpha):
    n, m = X.shape
    w = mp.Array(ctypes.c_double, m + 1)
    lock = Lock()
    procs = [Process(target=batch, args=(w, X, y, alpha, lock)) for i in range(1000)]
    for p in procs: 
        p.start()
    for p in procs: 
        p.join()

    return np.frombuffer(w.get_obj())

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
Xr, yr = X[indices,], y[indices,]
# train
w = train(Xr, yr, alpha=0.3)
p = [predict(w, x) for x in Xr]
print np.mean(yr == map(round, p))
