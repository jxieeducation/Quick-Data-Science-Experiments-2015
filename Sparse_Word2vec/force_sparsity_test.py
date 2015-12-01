from gensim.models.word2vec import Word2Vec
import numpy as np
from scipy.sparse import csr_matrix
import sys
import cPickle as pickle
from math import floor
from math import log10

f = open('t8.shakespeare.txt')
lines = f.readlines()
formatted_lines = []
for line in lines:
	line = line.replace('\n', '')
	if line:
		formatted_lines += [line.split()]

alpha_0 = 0.025 #can also start at 0.015
alpha_final = 0.00001
chunk_size = 1000

chunks=[formatted_lines[x:x+chunk_size] for x in xrange(0, len(formatted_lines), chunk_size)]
chunks = [formatted_lines] + [formatted_lines] + [formatted_lines] + [formatted_lines] + [formatted_lines] + [formatted_lines]
model = Word2Vec(formatted_lines, workers=8)
print model.most_similar("night")

def turn_to_sparse(matrix, rate=98):
	forced_sparse_arry = matrix.copy()
	cutoff = np.percentile(matrix, 98)
	forced_sparse_arry[forced_sparse_arry < cutoff] = 0
	return forced_sparse_arry

def retrain(chunks, model):
	alpha_decay_rate = (alpha_0 - alpha_final) / len(chunks)
	prev_0 = model.syn0.copy()
	prev_1 = model.syn1.copy()
	start_rate = 50
	end_rate = 80
	rate_decay_rate = (start_rate - end_rate) / len(chunks)
	for i in range(len(chunks)):
		model.alpha = alpha_0 - alpha_decay_rate * i
		model.min_alpha = alpha_0 - alpha_decay_rate * (i + 1)
		model.train(chunks[i])
		# 0 get gradient
		grad_0 = model.syn0 - prev_0
		grad_1 = model.syn1 - prev_1
		# 1 print stuff
		sparse_rate = start_rate - rate_decay_rate * i
		print sparse_rate
		print "original array size: " + str(sys.getsizeof(pickle.dumps(grad_0, -1)))
		grad_0 = turn_to_sparse(grad_0, sparse_rate)
		print "magnitude: " + str(np.sqrt(np.vdot(grad_0, grad_0)))
		sparse_grad_0 = csr_matrix(grad_0)
		print "modified sparse array size: " + str(sys.getsizeof(pickle.dumps(sparse_grad_0, -1)))
		print "\n"
		grad_1 = turn_to_sparse(grad_1, sparse_rate)
		# 2 apply gradient
		model.syn0 = prev_0 + grad_0
		model.syn1 = prev_1 + grad_1
		# end prepare for next iteration
		prev_0 = model.syn0.copy()
		prev_1 = model.syn1.copy()
	return model

model = retrain(chunks, model)
print model.most_similar("night")
