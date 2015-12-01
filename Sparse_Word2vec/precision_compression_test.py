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

def round_to_1(x):
	return round(x, -int(floor(log10(x))))

def roundNumpyArray(matrix):
	new_matrix = matrix.copy()
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if matrix[i][j] == 0:
				continue
			negative = False
			if matrix[i][j] < 0:
				negative = True
			rounded = round_to_1(abs(matrix[i][j]))
			if negative:
				new_matrix[i][j] = -rounded
			else:
				new_matrix[i][j] = rounded
	return new_matrix

def retrain(chunks, model):
	alpha_decay_rate = (alpha_0 - alpha_final) / len(chunks)
	prev_0 = model.syn0.copy()
	prev_1 = model.syn1.copy()
	for i in range(len(chunks)):
		model.alpha = alpha_0 - alpha_decay_rate * i
		model.min_alpha = alpha_0 - alpha_decay_rate * (i + 1)
		model.train(chunks[i])
		# 0 get gradient
		grad_0 = model.syn0 - prev_0
		grad_1 = model.syn1 - prev_1
		# 1 print stuff
		# print "normal array size: " + str(sys.getsizeof(pickle.dumps(grad_0, -1)))
		grad_0 = roundNumpyArray(grad_0)
		print model.alpha
		print grad_0
		print "magnitude: " + str(np.sqrt(np.vdot(grad_0, grad_0)))
		print "\n"
		grad_1 = roundNumpyArray(grad_1)
		# 2 apply gradient
		model.syn0 = prev_0 + grad_0
		model.syn1 = prev_1 + grad_1
		# end prepare for next iteration
		prev_0 = model.syn0.copy()
		prev_1 = model.syn1.copy()
	return model

model = retrain(chunks, model)
print model.most_similar("night")
