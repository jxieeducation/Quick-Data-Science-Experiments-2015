import numpy as np
from gensim.models import Word2Vec
import sys
from os import listdir
from os.path import isfile, join

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SEED = 1337
np.random.seed(SEED)

datadir = "data/cleaned-rap-lyrics/"
files = [join(datadir, f) for f in listdir(datadir) if isfile(join(datadir, f)) and ".txt" in f]
training_set = []
for rap_file in files:
	rap_file = open(rap_file)
	lines = rap_file.read().split('\n')
	cleaned_lines = [line.split(' ') for line in lines]
	training_set += cleaned_lines

print "finished loading corpus in memory"

model = Word2Vec(training_set, min_count=1)

model.save('data/rap.model')
