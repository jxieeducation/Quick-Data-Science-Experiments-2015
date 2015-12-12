import numpy as np
from gensim.models import Word2Vec
import cPickle as pickle


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SEED = 1337
np.random.seed(SEED)
words = pickle.load(open('data/munged_dataset.pickle'))

model = Word2Vec(words, min_count=10)

model.save('data/rap.model')
