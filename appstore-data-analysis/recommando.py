from gensim.models import Word2Vec
import sys

model = Word2Vec.load(sys.argv[1])

def recommando(positive=[], negative=[]):
	candidates = model.most_similar(positive=positive, negative=negative, topn=1000)
	top10 = []
	for word, score in candidates:
		if 'com.' in word or 'co.' in word:
			top10 += [word]
		if len(top10) == 10:
			break
	return top10
