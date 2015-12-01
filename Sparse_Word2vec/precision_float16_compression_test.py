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

# chunks=[formatted_lines[x:x+chunk_size] for x in xrange(0, len(formatted_lines), chunk_size)]
chunks = [formatted_lines] + [formatted_lines] + [formatted_lines] + [formatted_lines] + [formatted_lines] + [formatted_lines]
model = Word2Vec(formatted_lines, workers=8)
print model.most_similar("night")

def round_to_1(x):
	if x == 0:
		return x, 0
	if x > 0:
		return int(x * (10 ** -int(floor(log10(x))))), int(floor(log10(x)))
	else:
		return int(x * (10 ** -int(floor(log10(-x))))), int(floor(log10(-x)))

def retrain(chunks, model):
	alpha_decay_rate = (alpha_0 - alpha_final) / len(chunks)
	prev_0 = model.syn0.copy()
	prev_1 = model.syn1.copy()

	round_1 = np.vectorize(round_to_1)
	for i in range(len(chunks)):
		model.alpha = alpha_0 - alpha_decay_rate * i
		model.min_alpha = alpha_0 - alpha_decay_rate * (i + 1)
		model.train(chunks[i])
		# 0 get gradient
		grad_0 = model.syn0 - prev_0
		grad_1 = model.syn1 - prev_1
		# 2 apply gradient
		model.syn0 = prev_0 + grad_0
		model.syn1 = prev_1 + grad_1
		# 3 adhoc print
		print "normal array size: " + str(sys.getsizeof(pickle.dumps(grad_0, -1)))
		print grad_0
		grad_0 = grad_0.astype(np.float16)
		print grad_0
		print "new size: " + str(sys.getsizeof(pickle.dumps(grad_0, -1)))
		print "\n"
		# end prepare for next iteration
		prev_0 = model.syn0.copy()
		prev_1 = model.syn1.copy()
	return model

model = retrain(chunks, model)
print model.most_similar("night")
