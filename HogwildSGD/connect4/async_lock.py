import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.preprocessing import scale
from pandas import Series, DataFrame
import pandas as pd
import ctypes
import multiprocessing as mp
from multiprocessing import Process, Value, Lock
import time
import sys

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

def update(w, x, y, alpha, lock):
    new_w = np.frombuffer(w.get_obj())
    p = predict(new_w, x)
    with lock:
        new_w -= alpha * (p - y) * np.append(x, 1)

def batch(w, X, y, alpha, lock):
    for i in xrange(X.shape[0]):
        update(w, X[i,:], y[i], alpha, lock)
    
def train(X, y, alpha):
    total_sets = 10
    thread_num = int(sys.argv[1])

    n, m = X.shape
    w = mp.Array(ctypes.c_double, m + 1)
    lock = Lock()
    x_list = []
    y_list = []
    for i in range(thread_num):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        indices = indices[:max(total_sets * X.shape[0] / thread_num, X.shape[0])]
        x_list += [X[indices,]]
        y_list += [y[indices,]]
    procs = [Process(target=batch, args=(w, x_list[i], y_list[i], alpha, lock)) for i in range(thread_num)]
    start = time.time()
    for p in procs: 
        p.start()
    for p in procs: 
        p.join()
    end = time.time()
    print end - start
    return np.frombuffer(w.get_obj())

w = train(X, y, alpha=0.1)
p = [predict(w, x) for x in X]
print np.mean(y == map(round, p))
