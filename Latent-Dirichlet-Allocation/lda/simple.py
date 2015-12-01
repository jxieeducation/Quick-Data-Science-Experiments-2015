import numpy as np
import lda

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X)

topic_word = model.topic_word_
n_top_words = 8

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

