from gensim.models import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print "transforming data"
f = open('android_refined')
lines = f.read().split('\n')
corpus = [line.split() for line in lines]

print "starting word2vec"
model = Word2Vec(corpus, seed=1337, min_count=5, workers=4, window=20)
model.save('data/app_5_min.mod')
