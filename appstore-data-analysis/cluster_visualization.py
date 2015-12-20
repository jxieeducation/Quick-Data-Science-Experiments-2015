"""
arg1 - model location
arg2 - how many apps to include in space
"""
from gensim.models import Word2Vec
import sys
import operator
import numpy as np
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.metrics import euclidean_distances
import random

visualize_count = int(sys.argv[2])
if visualize_count < 500:
	visualize_count = 500

model = Word2Vec.load(sys.argv[1])
labels = []
vectors = np.array([np.zeros(model[model.vocab.keys()[0]].shape)])

print "selecting words"
word_count_dict = {}
for word in model.vocab.keys():
	if 'com.' in word or 'co.' in word:
		word_count_dict[word] = model.vocab[word].count
# goodPairs = sorted(word_count_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[:visualize_count]
# goodWords = [word for word,_ in goodPairs]
goodPairs = random.sample(list(word_count_dict), visualize_count)
goodWords = goodPairs
pairs = {'communication': ['com.facebook.katana', 'com.instagram.android', 'com.snapchat.android', 'com.twitter.android'],
	'minecraft': ['com.mojang.minecraftpe', 'com.plabs.planetofc', 'com.craftgames.worldcrft'],
	'voicegame': ['com.oki.phonenew', 'com.outfit7.talkingtom', 'au.com.penguinapps.android.babyphone'],
	'language': ['com.google.android.apps.translate', 'com.google.android.inputmethod.pinyin', 'com.croquis.biscuit']
}
apps_that_i_want = pairs['language']
goodWords = list(set(goodWords + apps_that_i_want))

print "constructing matrix for multi-dimension scaling"
for word in goodWords:
	labels += [word]
	vectors = np.concatenate((vectors, [model[word]]), axis=0)

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-2, random_state=1337, dissimilarity="euclidean", n_jobs=-1, verbose=5, n_init=3)
print "doing manifold fitting"
pos = mds.fit(vectors.astype(np.float32)).embedding_

print "graphing"
fig = plt.figure(1)
ax = plt.axes([0., 0., 1., 1.])
ax_ = fig.add_subplot(111)
plt.scatter(pos[:,0], pos[:,1], s=20, c='r')
for i in range(len(labels)):
	label = labels[i]
	if label in apps_that_i_want:
		xy = pos[i,]
		ax_.annotate(label, xy=xy)
plt.show()
