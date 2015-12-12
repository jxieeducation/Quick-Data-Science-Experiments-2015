import numpy as np
from gensim.models import Word2Vec


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SEED = 1337
np.random.seed(SEED)


# print "finished loading corpus in memory"

# model = Word2Vec(training_set, min_count=1)

# model.save('data/rap.model')
