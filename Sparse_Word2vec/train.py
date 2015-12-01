from gensim.models.word2vec import Word2Vec
import numpy as np
from scipy.sparse import csr_matrix
import sys
import cPickle as pickle

f = open('t8.shakespeare.txt')
lines = f.readlines()
formatted_lines = []
for line in lines:
	line = line.replace('\n', '')
	if line:
		formatted_lines += [line.split()]
initial_set = formatted_lines + formatted_lines + formatted_lines

chunk_size = 1000
chunks=[formatted_lines[x:x+chunk_size] for x in xrange(0, len(formatted_lines), chunk_size)]
fake_chunks = [formatted_lines] +  [formatted_lines] + [formatted_lines]
model = Word2Vec(formatted_lines, workers=8)
print model.most_similar("night")

def retrain(chunks, model):
	prev_0 = model.syn0.copy()
	for chunk in chunks:
		model.train(chunk)
		grad = model.syn0 - prev_0
		print "shape:" + str(grad.shape)
		print "magnitude:" + str(np.sqrt(np.vdot(grad, grad)))
		print "non-zero: " + str(np.count_nonzero(grad))
		print "mean: " + str(np.mean(grad))
		print "std: " + str(np.std(grad))
		print "50th: " + str(np.percentile(grad, 50))
		print "75th: " + str(np.percentile(grad, 75))
		print "90th: " + str(np.percentile(grad, 90))
		print "95th: " + str(np.percentile(grad, 95))
		print "normal array size: " + str(sys.getsizeof(pickle.dumps(grad, -1)))
		sparse_grad = csr_matrix(grad)
		print "sparse array size: " + str(sys.getsizeof(pickle.dumps(sparse_grad, -1)))
		forced_sparse_arry = grad.copy()
		forced_sparse_arry[forced_sparse_arry < np.percentile(grad, 75)] = 0
		forced_sparse_arry = csr_matrix(forced_sparse_arry)
		print "modified sparse array size: " + str(sys.getsizeof(pickle.dumps(forced_sparse_arry, -1)))
		print "\n"
		prev_0 = model.syn0.copy()
	return model

for i in range(3):
	model = retrain(chunks, model)
print model.most_similar("night")
