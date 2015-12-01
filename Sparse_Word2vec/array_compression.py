from gensim.models.word2vec import Word2Vec
import numpy as np
from scipy.sparse import csr_matrix
import sys
import cPickle as pickle
from math import floor
from math import log10

a = np.array([0.001, 20, 0.003, 0.4, 0.000000005])
print "original size: " + str(sys.getsizeof(pickle.dumps(a, -1)))

def round_to_1(x):
	if x == 0:
		return x, 0
	if x > 0:
		return int(x * (10 ** -int(floor(log10(x))))), int(floor(log10(x)))
	else:
		return int(x * (10 ** -int(floor(log10(-x))))), int(floor(log10(-x)))


round_1 = np.vectorize(round_to_1)
base, exponents = round_1(a)
base = base.astype(np.int8)
exponents = exponents.astype(np.int8)
	
print base
print exponents
print "new size: " + str(sys.getsizeof(pickle.dumps(base, -1)) + sys.getsizeof(pickle.dumps(exponents, -1)))
