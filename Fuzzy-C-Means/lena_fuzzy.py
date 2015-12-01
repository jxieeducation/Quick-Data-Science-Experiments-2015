from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy as sp
import numpy as np
import skfuzzy as fuzz

try:
	lena = sp.lena()
except AttributeError:
	from scipy import misc
	lena = misc.lena()

X = lena.reshape((-1, 1)).transpose()

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X, 5, 2, error=0.005, maxiter=1000, init=None)

def getCluster(pt, cntr):
	min_dist = 99999999
	min_cntr = 9999
	for cn in cntr:
		cn = cn[0]
		if abs(pt - cn) < min_dist:
			min_cntr = cn
			min_dist = abs(pt - cn)
	return min_cntr

print "compressing"
lena_compressed = X.reshape((-1, 1))
for i in range(lena_compressed.shape[0]):
	lena_compressed[i][0] = getCluster(lena_compressed[i][0], cntr)

lena_compressed.shape = lena.shape

plt.imshow(lena_compressed, cmap = cm.Greys_r)
plt.show()
