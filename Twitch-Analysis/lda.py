import logging
import itertools
from gensim import corpora, models, similarities
import numpy as np
import gensim
from collections import defaultdict
from pprint import pprint

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO 

lines = open('sample.txt').read().split('\n')

stoplist = set('for a of the and to in'.split())
texts = [[word for word in line.lower().split() if word not in stoplist] for line in lines]

frequency = defaultdict(int)
for text in texts:
	for token in text:
		frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(texts)
dictionary.save('kappa.dict')

lda = gensim.models.ldamodel.LdaModel(corpus=texts, id2word=dictionary, num_topics=10, update_every=1, chunksize=10000, passes=1)
