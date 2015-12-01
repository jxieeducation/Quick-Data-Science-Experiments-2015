import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import cluster, datasets
import scipy as sp
import numpy as np

try:
	lena = sp.lena()
except AttributeError:
	from scipy import misc
	lena = misc.lena()

X = lena.reshape((-1, 1)) # We need an (n_sample, n_feature) array

k_means = cluster.KMeans(n_clusters=5, n_init=1)
k_means.fit(X) 
values = k_means.cluster_centers_.squeeze()

labels = k_means.labels_
lena_compressed = np.choose(labels, values)
lena_compressed.shape = lena.shape

plt.imshow(lena_compressed, cmap = cm.Greys_r)
plt.show()
